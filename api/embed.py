from typing import List
import torch
from sentence_transformers import SentenceTransformer

# Ładowanie modelu lokalnie
# Możesz zmienić na większy np. "BAAI/bge-base-en-v1.5" lub "BAAI/bge-large-en-v1.5"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Wybór urządzenia: GPU (cuda) albo CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inicjalizacja modelu – tylko raz
_model = SentenceTransformer(MODEL_NAME, device=DEVICE)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Tworzy embeddingi dla listy tekstów.
    Zwraca listę wektorów (list[float]).
    """
    embeddings = _model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True  # normalizacja ułatwia porównania (cosine similarity)
    )
    return embeddings.tolist()

if __name__ == "__main__":
    # Test lokalny
    sample = ["Sztuczna inteligencja jest przyszłością.", "To jest test embeddowania."]
    vecs = embed_texts(sample)
    print(f"✅ Utworzono {len(vecs)} embeddingów, każdy o wymiarze {len(vecs[0])}")
