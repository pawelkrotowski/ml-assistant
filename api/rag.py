# ml-assistant/api/rag.py
# ------------------------------------------------------------
# RAG utilities:
#  - build_prompt(user, contexts, language="pl") -> OpenAI-style messages
#  - call_llm(messages, temperature=0.2, max_tokens=800) -> str
# Supports:
#  - Ollama (default) with streaming + retry
#  - Azure OpenAI Chat Completions
# Configuration via ENV:
#  LLM_PROVIDER=ollama|azure
#  # OLLAMA
#  OLLAMA_BASE_URL=http://ollama:11434
#  OLLAMA_MODEL=phi3:mini             (np. phi3:mini, qwen2.5:3b-instruct, llama3.1:8b)
#  OLLAMA_NUM_PREDICT=256             (max tokens to generate)
#  LLM_TIMEOUT=300                    (read timeout in seconds)
#  # AZURE
#  AZURE_OPENAI_ENDPOINT=https://<your>.openai.azure.com
#  AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
#  AZURE_OPENAI_API_KEY=***
#  AZURE_OPENAI_API_VERSION=2024-02-15-preview
#  # Misc
#  MAX_CONTEXT_CHARS=1600             (per chunk clamp)
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import asyncio
from typing import List, Dict, Sequence

import httpx


# --------- ENV / Config ---------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Timeouts
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))  # seconds

# Context clamp
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "1600"))


# --------- Helpers ---------
def _clamp(s: str, max_chars: int) -> str:
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip() + " …"
    return s


def _format_context(contexts: Sequence[str]) -> str:
    """
    Numeruje i łączy fragmenty kontekstu. Każdy chunk przycinamy do MAX_CONTEXT_CHARS,
    aby nie wysadzić promptu (zwłaszcza z dużymi PDF).
    """
    lines = []
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] {_clamp(c.strip(), MAX_CONTEXT_CHARS)}")
    return "\n\n".join(lines)


def build_prompt(user: str, contexts: Sequence[str], language: str = "pl") -> List[Dict[str, str]]:
    """
    Buduje wiadomości w formacie Chat Completions (messages=[{role, content}, ...]).
    Zasada: odpowiadaj TYLKO na podstawie dostarczonego kontekstu; cytuj [#].
    """
    system_msg = (
        f"Jesteś asystentem RAG. Odpowiadasz krótko i rzeczowo po {language}. "
        "Korzystaj WYŁĄCZNIE z dostarczonego kontekstu poniżej. "
        "Jeśli kontekst nie zawiera odpowiedzi, powiedz wprost, że nie masz danych. "
        "Cytuj źródła poprzez odwołania [1], [2], … zgodnie z numeracją fragmentów."
    )
    context_block = _format_context(list(contexts))

    user_msg = (
        "KONTEKST:\n"
        f"{context_block}\n\n"
        "PYTANIE:\n"
        f"{user.strip()}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# --------- LLM calls ---------
async def _ollama_chat_stream(messages: List[Dict[str, str]], temperature: float) -> str:
    """
    Wywołuje Ollamę ze strumieniowaniem i scala odpowiedź.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": OLLAMA_NUM_PREDICT,
            # "num_ctx": 4096,  # odkomentuj, jeśli potrzebujesz większego kontekstu
        },
    }
    timeout = httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=60.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as r:
            r.raise_for_status()
            buf: List[str] = []
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    chunk = json.loads(line[5:].strip())
                except Exception:
                    continue
                # typowe pola Ollamy w streamie:
                msg = chunk.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    buf.append(msg["content"])
                elif "response" in chunk:
                    buf.append(chunk["response"])
            return "".join(buf).strip()


async def _azure_chat(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    """
    Azure OpenAI Chat Completions.
    """
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY):
        raise RuntimeError("Brak konfiguracji Azure OpenAI (endpoint/deployment/api key).")

    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    timeout = httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=60.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


async def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """
    Główna funkcja wywołania LLM zgodnie z LLM_PROVIDER.
    Dla Ollamy: streaming + 1 retry przy timeout/rozłączeniu.
    """
    if LLM_PROVIDER == "ollama":
        try:
            return await _ollama_chat_stream(messages, temperature)
        except (httpx.TimeoutException, httpx.ReadError) as e:
            # krótki retry (częste przy cold-start)
            await asyncio.sleep(1.0)
            return await _ollama_chat_stream(messages, temperature)
        except httpx.HTTPStatusError as e:
            # lepszy komunikat w logach
            body = e.response.text if e.response is not None else ""
            raise RuntimeError(f"Ollama HTTP {e.response.status_code if e.response else '??'}: {body}") from e

    elif LLM_PROVIDER == "azure":
        try:
            return await _azure_chat(messages, temperature, max_tokens)
        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            raise RuntimeError(f"Azure OpenAI HTTP {e.response.status_code if e.response else '??'}: {body}") from e

    else:
        raise ValueError(f"Nieobsługiwany LLM_PROVIDER: {LLM_PROVIDER!r}")


# --------- (Opcjonalnie) MMR re-ranking ---------
# Jeśli chcesz używać re-rankingu w main.py, możesz wywołać mmr(query, docs, k).
# Wymaga, by embed.embed_texts zwracało znormalizowane wektory (u nas tak jest).
try:
    from .embed import embed_texts  # noqa: WPS433
    import numpy as _np

    def mmr(query: str, docs: Sequence[str], k: int = 6, lambda_mult: float = 0.5) -> List[int]:
        """
        Maximal Marginal Relevance – wybiera k zróżnicowanych kontekstów.
        Zwraca indeksy wybranych dokumentów w kolejności.
        """
        if not docs:
            return []
        qv = _np.array(embed_texts([query])[0])
        dvs = _np.array(embed_texts(list(docs)))
        selected: List[int] = []
        candidates = set(range(len(docs)))

        def cos(a, b) -> float:
            # przy założeniu normalizacji w embed_texts dot == cosine
            return float(_np.dot(a, b))

        while candidates and len(selected) < min(k, len(docs)):
            best_score = None
            best_idx = None
            for i in candidates:
                sim_to_q = cos(qv, dvs[i])
                sim_to_sel = max((cos(dvs[i], dvs[j]) for j in selected), default=0.0)
                score = lambda_mult * sim_to_q - (1 - lambda_mult) * sim_to_sel
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)  # type: ignore[arg-type]
            candidates.remove(best_idx)  # type: ignore[arg-type]
        return selected

except Exception:
    # mmr opcjonalny – jeśli embed_texts nie jest dostępne na etapie importu
    def mmr(query: str, docs: Sequence[str], k: int = 6, lambda_mult: float = 0.5) -> List[int]:
        return list(range(min(k, len(docs))))


# --------- Local test ---------
if __name__ == "__main__":
    # Szybki suchy test budowy wiadomości (bez wołania LLM)
    demo_contexts = [
        "Dokument opisuje system RAG i pipeline: ingest → embed → search → chat.",
        "Używany jest Postgres z pgvector oraz FastAPI po stronie API.",
    ]
    msgs = build_prompt("Jak wygląda przepływ danych?", demo_contexts, language="pl")
    print("Messages:")
    for m in msgs:
        print(f"- {m['role']}: {m['content'][:120]}{'...' if len(m['content'])>120 else ''}")
    print("\nLLM_PROVIDER:", LLM_PROVIDER, "| OLLAMA_MODEL:", OLLAMA_MODEL)
