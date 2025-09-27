# ml-assistant/api/main.py
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.orm import aliased

from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from sqlalchemy import select, func

from .db import SessionLocal, init_db, Document, Chunk
from .chunking import chunk_any
from .embed import embed_texts
from .rag import build_prompt, call_llm

app = FastAPI(title="ML Assistant RAG API", version="0.2.0")

app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# CORS — ułatwia testy lokalne/GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DeleteDocsIn(BaseModel):
    ids: Optional[List[str]] = None        # UUID string(s) dokumentów
    sources: Optional[List[str]] = None    # np. nazwy plików (Document.source)
    delete_all: Optional[bool] = False     # hard reset wszystkiego


# Tworzenie tabel/indeksów przy starcie
init_db()


class ChatIn(BaseModel):
    message: str
    k: Optional[int] = 6
    language: Optional[str] = "en"

@app.delete("/documents")
def delete_documents(payload: DeleteDocsIn):
    if not payload.delete_all and not (payload.ids or payload.sources):
        # nic nie podano → nic nie robimy
        raise HTTPException(
            status_code=400,
            detail="Provide 'ids' or 'sources' or set 'delete_all': true."
        )

    with SessionLocal() as s:
        try:
            # ZBIERZ ID DO USUNIĘCIA
            ids_to_delete = set()

            if payload.delete_all:
                # weź wszystkie ID
                all_ids = s.execute(select(Document.id)).scalars().all()
                ids_to_delete.update(all_ids)
            else:
                if payload.ids:
                    ids_to_delete.update(payload.ids)
                if payload.sources:
                    src_ids = s.execute(
                        select(Document.id).where(Document.source.in_(payload.sources))
                    ).scalars().all()
                    ids_to_delete.update(src_ids)

            if not ids_to_delete:
                return {"deleted_documents": 0, "deleted_chunks": 0}

            # USUŃ CHUNKI → potem DOKUMENTY (w tej kolejności)
            del_chunks = s.execute(
                Chunk.__table__.delete().where(Chunk.document_id.in_(ids_to_delete))
            ).rowcount or 0

            del_docs = s.execute(
                Document.__table__.delete().where(Document.id.in_(ids_to_delete))
            ).rowcount or 0

            s.commit()
            return {"deleted_documents": int(del_docs), "deleted_chunks": int(del_chunks)}

        except Exception as e:
            s.rollback()
            raise HTTPException(status_code=500, detail=f"Delete error: {e}")


# --- USUWANIE POJEDYNCZEGO DOKUMENTU PO ID (opcjonalny, wygodny) ---
@app.delete("/documents/{doc_id}")
def delete_document_by_id(doc_id: str):
    with SessionLocal() as s:
        try:
            del_chunks = s.execute(
                Chunk.__table__.delete().where(Chunk.document_id == doc_id)
            ).rowcount or 0
            del_docs = s.execute(
                Document.__table__.delete().where(Document.id == doc_id)
            ).rowcount or 0
            s.commit()
            if del_docs == 0:
                raise HTTPException(status_code=404, detail="Document not found.")
            return {"deleted_documents": int(del_docs), "deleted_chunks": int(del_chunks)}
        except HTTPException:
            raise
        except Exception as e:
            s.rollback()
            raise HTTPException(status_code=500, detail=f"Delete error: {e}")

@app.get("/documents")
def list_documents():
    with SessionLocal() as s:
        d = aliased(Document)
        c = aliased(Chunk)
        rows = s.execute(
            select(
                d.id,
                d.source,
                func.count(c.id).label("chunks")
            ).outerjoin(c, c.document_id == d.id)
             .group_by(d.id, d.source)
             .order_by(d.source.asc())
        ).all()
        return [
            {"id": str(r.id), "source": r.source, "chunks": int(r.chunks)}
            for r in rows
        ]

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    with SessionLocal() as s:
        docs = s.execute(select(func.count()).select_from(Document)).scalar_one()
        chks = s.execute(select(func.count()).select_from(Chunk)).scalar_one()
    return {"documents": int(docs), "chunks": int(chks)}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload jednego pliku (PDF/TXT/MD). Chunkowanie -> embeddingi -> zapis do DB.
    """
    try:
        raw = await file.read()
        texts = chunk_any(raw, filename=file.filename)  # List[str]
        if not texts:
            raise HTTPException(status_code=400, detail="No content to ingest")

        vecs = embed_texts(texts)  # List[List[float]]

        with SessionLocal() as s:
            doc = Document(source=file.filename)
            s.add(doc)
            s.flush()  # doc.id

            for i, (t, v) in enumerate(zip(texts, vecs)):
                s.add(Chunk(document_id=doc.id, chunk_index=i, content=t, embedding=v))

            s.commit()

        return {"ok": True, "file": file.filename, "chunks": len(texts)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")


@app.post("/chat")
async def chat(payload: ChatIn):
    """
    Prosty RAG:
    - embed zapytania
    - wektorowe top-k z pgvector
    - prompt + wywołanie LLM
    - ZWRACA TYLKO 'answer' (bez kontekstów i bez cytatów)
    """
    q = (payload.message or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="message is empty")

    try:
        # 0) upewnij się, że są jakiekolwiek chunki
        with SessionLocal() as s:
            total_chunks = s.execute(select(func.count()).select_from(Chunk)).scalar_one()
        if total_chunks == 0:
            raise HTTPException(status_code=400, detail="No data ingested yet. Upload a file via /ingest first.")

        # 1) embedding zapytania
        q_vec = embed_texts([q])[0]
        k = int(payload.k or 6)

        # 2) wektorowe top-k (wewnętrznie; nie zwracamy na zewnątrz)
        with SessionLocal() as s:
            selector = getattr(Chunk, "select_contexts_by_embedding", None)
            if callable(selector):
                rows = s.execute(selector(q_vec, limit=k)).fetchall()
                contexts: List[str] = [r[2] for r in rows]  # tylko treść
            else:
                rows = s.execute(Chunk.select_content_by_embedding(q_vec, limit=k)).fetchall()
                contexts = [r[0] for r in rows]

        if not contexts:
            raise HTTPException(status_code=404, detail="No matching context found for this question.")

        # 3) budowa promptu + wywołanie LLM (citations=False)
        msgs = build_prompt(
            user=q,
            contexts=contexts,
            language=payload.language or "pl",
            citations=False,
        )
        answer = await call_llm(msgs)
        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")
