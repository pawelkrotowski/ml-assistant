# ml-assistant/api/main.py
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from .db import SessionLocal, init_db, Document, Chunk
from .chunking import chunk_any
from .embed import embed_texts
from .rag import build_prompt, call_llm


app = FastAPI(title="ML Assistant RAG API", version="0.1.1")

# CORS – ułatwia lokalne testy z przeglądarki
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tworzenie tabel/indeksów przy starcie
init_db()


class ChatIn(BaseModel):
    message: str
    k: Optional[int] = 6
    language: Optional[str] = "pl"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload jednego pliku (PDF/TXT/MD). Chunkowanie -> embeddingi -> zapis do DB.
    """
    try:
        raw = await file.read()
        texts = chunk_any(raw, filename=file.filename)  # List[str]
        if not texts:
            raise HTTPException(status_code=400, detail="Brak treści do przetworzenia")

        vecs = embed_texts(texts)  # List[List[float]]

        with SessionLocal() as s:
            doc = Document(source=file.filename)
            s.add(doc)
            s.flush()  # uzyskaj doc.id

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
    - zwrot odpowiedzi + konteksty (źródło i indeks chunku, jeśli dostępne)
    """
    q = (payload.message or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="message is empty")

    try:
        # 1) embedding zapytania
        q_vec = embed_texts([q])[0]
        k = int(payload.k or 6)

        # 2) wektorowe top-k
        with SessionLocal() as s:
            # Preferowany wariant (źródło + indeks):
            selector = getattr(Chunk, "select_contexts_by_embedding", None)
            if callable(selector):
                rows = s.execute(selector(q_vec, limit=k)).fetchall()
                contexts = [{"source": r[0], "index": r[1], "content": r[2]} for r in rows]
            else:
                # Wstecznie kompatybilny wariant (tylko content)
                rows = s.execute(Chunk.select_content_by_embedding(q_vec, limit=k)).fetchall()
                contexts = [{"source": "unknown", "index": i, "content": r[0]} for i, r in enumerate(rows)]

        # 3) budowa promptu + call LLM
        msgs = build_prompt(
            user=q,
            contexts=[c["content"] for c in contexts],
            language=payload.language or "pl",
        )
        answer = await call_llm(msgs)

        # 4) odpowiedź API
        return {"answer": answer, "contexts": contexts}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")


# (opcjonalnie) prosty endpoint diagnostyczny – ile dokumentów/chunków w bazie
@app.get("/stats")
def stats():
    with SessionLocal() as s:
        docs = s.execute("SELECT COUNT(*) FROM documents").scalar()
        chks = s.execute("SELECT COUNT(*) FROM chunks").scalar()
    return {"documents": int(docs or 0), "chunks": int(chks or 0)}
