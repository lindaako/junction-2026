"""Offline checks. Run from the backend folder:  python tests/test_basics.py
No pytest, no network and no API keys needed."""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import db  # noqa: E402
from app.providers import (  # noqa: E402
    ChatRequest, ProviderError, build_anthropic_payload, build_gemini_payload,
    build_openai_payload, call_model, parse_anthropic_response,
    parse_gemini_response, parse_openai_response,
)


def test_openai_payload_and_parse():
    req = ChatRequest(message="hi", system="be brief", history=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    p = build_openai_payload(req, "m")
    assert [m["role"] for m in p["messages"]] == ["system", "user", "assistant", "user"]
    assert p["messages"][-1]["content"] == "hi"
    assert parse_openai_response({"choices": [{"message": {"content": "yo"}}]}) == "yo"


def test_openai_image():
    req = ChatRequest(message="what is this", image_b64="AAAA", image_mime="image/jpeg")
    content = build_openai_payload(req, "m")["messages"][-1]["content"]
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"


def test_gemini():
    req = ChatRequest(message="hi", system="sys", image_b64="AAAA", history=[{"role": "assistant", "content": "b"}])
    p = build_gemini_payload(req)
    assert p["systemInstruction"]["parts"][0]["text"] == "sys"
    assert p["contents"][0]["role"] == "model"
    assert p["contents"][-1]["parts"][1]["inlineData"]["data"] == "AAAA"
    assert parse_gemini_response({"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}) == "ab"


def test_anthropic():
    req = ChatRequest(message="hi", system="sys")
    p = build_anthropic_payload(req, "m")
    assert p["system"] == "sys" and p["messages"][-1]["content"] == "hi"
    assert parse_anthropic_response({"content": [{"type": "text", "text": "x"}]}) == "x"


def test_bad_response_raises():
    try:
        parse_openai_response({"nope": 1})
    except ProviderError:
        return
    raise AssertionError("expected ProviderError")


def test_echo_and_unknown_provider():
    assert asyncio.run(call_model(ChatRequest(message="ping"), "echo")) == "[echo] ping"
    try:
        asyncio.run(call_model(ChatRequest(message="x"), "nope"))
    except ProviderError:
        return
    raise AssertionError("expected ProviderError")


def test_missing_key_message():
    os.environ.pop("GEMINI_API_KEY", None)
    try:
        asyncio.run(call_model(ChatRequest(message="x"), "gemini"))
    except ProviderError as e:
        assert "API key" in str(e)
        return
    raise AssertionError("expected ProviderError")


def test_db_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        db.SQLITE_PATH = os.path.join(d, "t.db")
        os.environ.pop("DATABASE_URL", None)
        db.init()
        db.save("echo", "p", "r", 12)
        rows = db.recent()
        assert len(rows) == 1 and rows[0]["prompt"] == "p" and rows[0]["latency_ms"] == 12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} passed")
