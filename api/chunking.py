from typing import List, Optional
import os

def _decode_bytes(raw: bytes) -> str:
    # Proste dekodowanie – ignoruje błędy
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

def _extract_pdf(raw: bytes) -> str:
    # Minimalny parser PDF (tekst) – bez obrazów/tabel
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("Zainstaluj pypdf (requirements.txt), aby przetwarzać PDF.")
    import io
    reader = PdfReader(io.BytesIO(raw))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def _split_with_overlap(text: str, chunk_size: int = 2000, overlap: int = 300) -> List[str]:
    texts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            texts.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return texts

def chunk_any(raw: bytes, filename: Optional[str] = None) -> List[str]:
    """
    Przyjmuje bajty i opcjonalnie nazwę pliku; zwraca listę chunków tekstu.
    Obsługa: .pdf (pypdf), pozostałe traktowane jako tekst.
    """
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        text = _extract_pdf(raw)
    else:
        text = _decode_bytes(raw)

    # Minimalne czyszczenie
    text = text.replace("\x00", " ").strip()

    # Chunkowanie
    return _split_with_overlap(text, chunk_size=2000, overlap=300)
