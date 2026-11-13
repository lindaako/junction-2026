import { useEffect, useState } from "react";
import { chat, getHistory, getProviders, type HistoryRow } from "./api";

export default function App() {
  const [providers, setProviders] = useState<Record<string, boolean>>({ echo: true });
  const [provider, setProvider] = useState("echo");
  const [message, setMessage] = useState("");
  const [image, setImage] = useState<{ b64: string; mime: string } | null>(null);
  const [reply, setReply] = useState("");
  const [meta, setMeta] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<HistoryRow[]>([]);

  const refreshHistory = () => getHistory().then(setHistory).catch(() => {});

  useEffect(() => {
    getProviders().then(setProviders).catch(() => setError("Backend not reachable. Is it running on :8000?"));
    refreshHistory();
  }, []);

  function onFile(file?: File) {
    if (!file) return setImage(null);
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result);
      setImage({ b64: dataUrl.split(",")[1], mime: file.type || "image/png" });
    };
    reader.readAsDataURL(file);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!message.trim()) return;
    setBusy(true);
    setError("");
    setReply("");
    try {
      const res = await chat(message, provider, image?.b64, image?.mime);
      setReply(res.reply);
      setMeta(`${res.provider} · ${res.latency_ms} ms`);
      refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header>
        <p className="eyebrow">Junction 2026</p>
        <h1>Your project name</h1>
        <p className="lede">One sentence on what it does and who it is for.</p>
      </header>

      <form className="card" onSubmit={submit}>
        <label>
          Model
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            {Object.entries(providers).map(([name, ok]) => (
              <option key={name} value={name} disabled={!ok}>
                {name}
                {ok ? "" : " (no key)"}
              </option>
            ))}
          </select>
        </label>
        <textarea
          rows={4}
          placeholder="Ask something, or describe the input…"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
        <div className="row">
          <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} />
          <button disabled={busy || !message.trim()}>{busy ? "Thinking…" : "Run"}</button>
        </div>
      </form>

      {error && <p className="error">{error}</p>}
      {reply && (
        <section className="card result">
          <p className="meta">{meta}</p>
          <p>{reply}</p>
        </section>
      )}

      {history.length > 0 && (
        <section className="history">
          <h2>Recent</h2>
          {history.map((h) => (
            <details key={h.id}>
              <summary>{h.prompt.slice(0, 70)}</summary>
              <p>{h.reply}</p>
              <p className="meta">{h.provider} · {h.latency_ms} ms</p>
            </details>
          ))}
        </section>
      )}
    </main>
  );
}
