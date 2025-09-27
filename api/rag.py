# ml-assistant/api/rag.py
# ------------------------------------------------------------
# RAG utilities:
#  - build_prompt(user, contexts, language="pl", citations=False) -> messages
#  - call_llm(messages, temperature=0.2, max_tokens=800, model=None, num_ctx=None, extra_options=None) -> str
# Providers:
#  - Ollama (default) with streaming + retry
#  - Azure OpenAI (Chat Completions)
# ENV:
#  LLM_PROVIDER=ollama|azure
#  OLLAMA_BASE_URL=http://ollama:11434
#  OLLAMA_MODEL=qwen2.5:7b-instruct      # np. qwen2.5:7b-instruct, qwen2.5:14b-instruct, llama3.1:8b
#  OLLAMA_NUM_PREDICT=256
#  OLLAMA_NUM_CTX=8192
#  OLLAMA_OPTIONS_JSON={}                # np. {"mirostat":1,"mirostat_tau":5}
#  LLM_TIMEOUT=300                       # sekundy
#  AZURE_OPENAI_ENDPOINT=...
#  AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
#  AZURE_OPENAI_API_KEY=...
#  AZURE_OPENAI_API_VERSION=2024-02-15-preview
#  MAX_CONTEXT_CHARS=1600
# ------------------------------------------------------------

from __future__ import annotations

import os
import json
import asyncio
from typing import List, Dict, Sequence, Optional, Any

import httpx

FALLBACK_EMPTY = "Nie mam wystarczających informacji w dostarczonym kontekście."

# --- Konfiguracja z ENV ---
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
try:
    OLLAMA_OPTIONS_JSON = json.loads(os.environ.get("OLLAMA_OPTIONS_JSON", "{}") or "{}")
except json.JSONDecodeError:
    OLLAMA_OPTIONS_JSON = {}

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Timeouty
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))

# Przycinanie kontekstu
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "1600"))


# --- Helpery ---
def _clamp(s: str, max_chars: int) -> str:
    s = s.strip()
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars].rstrip() + " …"
    return s


def _format_context(contexts: Sequence[str], use_citations: bool) -> str:
    # U Ciebie chcemy bez cytatów, więc domyślnie 'use_citations=False'
    lines = []
    for i, c in enumerate(contexts, start=1):
        c = _clamp(c, MAX_CONTEXT_CHARS)
        lines.append(f"[{i}] {c}" if use_citations else c)
    return "\n\n".join(lines)


def build_prompt(
    user: str,
    contexts: Sequence[str],
    language: str = "pl",
    citations: bool = False,  # domyślnie wyłączone
) -> List[Dict[str, str]]:
    system_msg = (
        f"Jesteś asystentem RAG. Odpowiadasz krótko i rzeczowo po {language}. "
        "Korzystaj WYŁĄCZNIE z dostarczonego kontekstu. "
        "Jeśli kontekst nie zawiera odpowiedzi, napisz dokładnie: "
        "\"Nie mam wystarczających informacji w dostarczonym kontekście.\" "
        "Nie używaj cytowań ani nawiasów w stylu [n]."
    )
    context_block = _format_context(list(contexts), use_citations=citations)
    user_msg = f"KONTEKST:\n{context_block}\n\nPYTANIE:\n{user.strip()}"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# --- OLLAMA ---
async def _ollama_chat_stream(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float,
    num_predict: int,
    num_ctx: int,
    extra_options: Optional[Dict[str, Any]] = None,
) -> str:
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }
    if OLLAMA_OPTIONS_JSON:
        options.update(OLLAMA_OPTIONS_JSON)
    if extra_options:
        options.update(extra_options)

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    timeout = httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=60.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as r:
            r.raise_for_status()
            buf: List[str] = []
            async for line in r.aiter_lines():
                if not line:
                    continue
                # Ollama streamuje czysty JSON per linia; czasem prefiks "data:"
                raw = line[5:].strip() if line.startswith("data:") else line
                try:
                    chunk = json.loads(raw)
                except Exception:
                    continue
                msg = chunk.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    buf.append(msg["content"])
                elif "response" in chunk:  # alternatywny format
                    buf.append(chunk["response"])
            return "".join(buf).strip()


async def _ollama_chat_nostream(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float,
    num_predict: int,
    num_ctx: int,
    extra_options: Optional[Dict[str, Any]] = None,
) -> str:
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }
    if OLLAMA_OPTIONS_JSON:
        options.update(OLLAMA_OPTIONS_JSON)
    if extra_options:
        options.update(extra_options)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    timeout = httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=60.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return (data["message"].get("content") or "").strip()
            if "choices" in data and data["choices"]:
                return (data["choices"][0]["message"]["content"] or "").strip()
        return ""


# --- AZURE OPENAI ---
async def _azure_chat(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY):
        raise RuntimeError("Azure OpenAI not configured (endpoint/deployment/api key).")

    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    timeout = httpx.Timeout(connect=10.0, read=LLM_TIMEOUT, write=60.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


# --- PUBLIC API ---
async def call_llm(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 800,
    model: Optional[str] = None,           # <— per-request override
    num_ctx: Optional[int] = None,         # <— per-request override
    extra_options: Optional[Dict[str, Any]] = None,  # dodatkowe opcje do Ollamy
) -> str:
    """
    Wywołuje LLM zgodnie z LLM_PROVIDER.
    - Dla Ollamy: używa modelu z parametru, ENV OLLAMA_MODEL lub domyślnie qwen2.5:7b-instruct.
    - Dla Azure: używa ustawionego deploymentu.
    """
    if LLM_PROVIDER == "ollama":
        mdl = (model or OLLAMA_MODEL).strip()
        ctx = int(num_ctx or OLLAMA_NUM_CTX)
        # 1) streaming z retry
        try:
            out = await _ollama_chat_stream(
                messages,
                model=mdl,
                temperature=temperature,
                num_predict=OLLAMA_NUM_PREDICT,
                num_ctx=ctx,
                extra_options=extra_options,
            )
            if out:
                return out
        except (httpx.TimeoutException, httpx.ReadError):
            await asyncio.sleep(0.5)  # drobny retry delay
        # 2) fallback bez streamu
        try:
            out2 = await _ollama_chat_nostream(
                messages,
                model=mdl,
                temperature=temperature,
                num_predict=OLLAMA_NUM_PREDICT,
                num_ctx=ctx,
                extra_options=extra_options,
            )
            return out2 or FALLBACK_EMPTY
        except Exception:
            return FALLBACK_EMPTY

    elif LLM_PROVIDER == "azure":
        return await _azure_chat(messages, temperature, max_tokens)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}")
