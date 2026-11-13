"""One function, call_model(), that can talk to any model backend.

Switch backends with the PROVIDER env var (or per request) - nothing else in
the app changes. The request builders / response parsers are pure functions so
they can be tested without network access.

Providers:
  echo       no key needed, returns your prompt (use to test the pipeline)
  gemini     Google AI Studio key
  openai     OpenAI API key
  anthropic  Anthropic API key
  local      ANY OpenAI-compatible server: Ollama, vLLM on Modal/Verda,
             Featherless, LM Studio, ... (set LOCAL_BASE_URL / LOCAL_MODEL)
"""
from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    pass


@dataclass
class ChatRequest:
    message: str
    system: str | None = None
    image_b64: str | None = None  # raw base64, no "data:" prefix
    image_mime: str = "image/png"
    history: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": "..."}]


# ---------------------------------------------------------------- config

def _cfg(provider: str) -> dict:
    env = os.getenv
    return {
        "gemini": {
            "key": env("GEMINI_API_KEY", ""),
            "model": env("GEMINI_MODEL", "gemini-3.8-flash"),
        },
        "openai": {
            "key": env("OPENAI_API_KEY", ""),
            "model": env("OPENAI_MODEL", "gpt-4o-mini"),
            "base": "https://api.openai.com/v1",
        },
        "anthropic": {
            "key": env("ANTHROPIC_API_KEY", ""),
            "model": env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        },
        "local": {
            "key": env("LOCAL_API_KEY", "not-needed"),
            "model": env("LOCAL_MODEL", "llama3.1:8b"),
            "base": env("LOCAL_BASE_URL", "http://localhost:11434/v1"),
        },
        "echo": {},
    }[provider]


def available_providers() -> dict[str, bool]:
    """Provider name -> is it configured (has a key / is reachable by config)."""
    return {
        "echo": True,
        "gemini": bool(_cfg("gemini")["key"]),
        "openai": bool(_cfg("openai")["key"]),
        "anthropic": bool(_cfg("anthropic")["key"]),
        "local": bool(os.getenv("LOCAL_BASE_URL")),
    }


# ------------------------------------------------- pure request/response code

def build_openai_payload(req: ChatRequest, model: str) -> dict:
    messages: list[dict] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend(req.history)
    if req.image_b64:
        content = [
            {"type": "text", "text": req.message},
            {"type": "image_url",
             "image_url": {"url": f"data:{req.image_mime};base64,{req.image_b64}"}},
        ]
    else:
        content = req.message
    messages.append({"role": "user", "content": content})
    return {"model": model, "messages": messages}


def parse_openai_response(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"Unexpected OpenAI-style response: {str(data)[:300]}") from e


def build_gemini_payload(req: ChatRequest) -> dict:
    contents = []
    for turn in req.history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    parts: list[dict] = [{"text": req.message}]
    if req.image_b64:
        parts.append({"inlineData": {"mimeType": req.image_mime, "data": req.image_b64}})
    contents.append({"role": "user", "parts": parts})
    payload: dict = {"contents": contents}
    if req.system:
        payload["systemInstruction"] = {"parts": [{"text": req.system}]}
    return payload


def parse_gemini_response(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"Unexpected Gemini response: {str(data)[:300]}") from e


def build_anthropic_payload(req: ChatRequest, model: str) -> dict:
    messages = [{"role": t["role"], "content": t["content"]} for t in req.history]
    if req.image_b64:
        content = [
            {"type": "image",
             "source": {"type": "base64", "media_type": req.image_mime, "data": req.image_b64}},
            {"type": "text", "text": req.message},
        ]
    else:
        content = req.message
    messages.append({"role": "user", "content": content})
    payload: dict = {"model": model, "max_tokens": 1024, "messages": messages}
    if req.system:
        payload["system"] = req.system
    return payload


def parse_anthropic_response(data: dict) -> str:
    try:
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
    except (KeyError, TypeError) as e:
        raise ProviderError(f"Unexpected Anthropic response: {str(data)[:300]}") from e


# ------------------------------------------------------------- the one entry point

async def call_model(req: ChatRequest, provider: str | None = None) -> str:
    name = (provider or os.getenv("PROVIDER", "echo")).lower()
    if name == "echo":
        return f"[echo] {req.message}"
    if name not in ("gemini", "openai", "anthropic", "local"):
        raise ProviderError(f"Unknown provider '{name}'")

    cfg = _cfg(name)
    if name != "local" and not cfg["key"]:
        raise ProviderError(f"Provider '{name}' selected but its API key is not set in .env")

    import httpx  # imported lazily so the pure functions above work without it

    if name == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent"
        headers = {"x-goog-api-key": cfg["key"]}
        payload, parse = build_gemini_payload(req), parse_gemini_response
    elif name == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": cfg["key"], "anthropic-version": "2023-06-01"}
        payload, parse = build_anthropic_payload(req, cfg["model"]), parse_anthropic_response
    else:  # openai or local: both speak the OpenAI chat-completions format
        url = cfg["base"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {cfg['key']}"}
        payload, parse = build_openai_payload(req, cfg["model"]), parse_openai_response

    r = None
    last_err = None
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(4):
            try:
                r = await client.post(url, json=payload, headers=headers)
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
            except httpx.HTTPError as e:
                last_err = e
                r = None
            if attempt < 3:
                await asyncio.sleep(1.5 * (attempt + 1))
    if r is None:
        raise ProviderError(f"Could not reach {name}: {last_err}")

    if r.status_code >= 400:
        raise ProviderError(f"{name} returned {r.status_code}: {r.text[:300]}")
    return parse(r.json())
