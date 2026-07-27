import { useEffect, useRef, useState } from "react";
import { api, type ChatMessage } from "../api";
import { Button, ErrorNote } from "./shell";

const SUGGESTIONS = [
  "What's in this data?",
  "Which columns have missing values?",
  "How did the model do?",
];

export function Chat({ projectId, hasDataset }: { projectId: string; hasDataset: boolean }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.chatHistory(projectId).then(setMessages).catch(() => setMessages([]));
  }, [projectId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages.length, busy]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setBusy(true);
    setError(null);
    setDraft("");
    // Show the question immediately; the server persists its own copy.
    setMessages((m) => [
      ...m,
      { id: `local-${Date.now()}`, role: "user", content: question, created_at: "" },
    ]);
    try {
      const reply = await api.chatAsk(projectId, question);
      setMessages((m) => [...m, reply]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="flex h-[480px] flex-col rounded-md border border-rule bg-surface">
      <header className="flex items-baseline justify-between border-b border-rule px-4 py-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-faint">
          Ask about this data
        </h2>
        {messages.length > 0 && (
          <button
            onClick={() => api.chatClear(projectId).then(() => setMessages([]))}
            className="text-[12px] text-ink-faint hover:text-ink"
          >
            clear
          </button>
        )}
      </header>

      <div className="flex-1 space-y-3.5 overflow-y-auto px-4 py-3.5">
        {messages.length === 0 && (
          <div>
            <p className="text-[13px] leading-relaxed text-ink-soft">
              Questions are answered by computing real values from your table —
              never estimated. Try one of these:
            </p>
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  disabled={!hasDataset}
                  className="rounded border border-rule-strong px-2 py-1 text-[12px] text-ink-soft hover:bg-paper disabled:opacity-40"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? "text-right" : ""}>
            <div
              className={`inline-block max-w-[92%] rounded-md px-3 py-2 text-left text-[13px] leading-relaxed ${
                m.role === "user"
                  ? "bg-verified-wash text-ink"
                  : "border border-rule bg-paper text-ink-soft"
              }`}
            >
              {m.content}
            </div>
            {!!m.metadata?.tools?.length && (
              <div className="tabular mt-1 text-[11px] text-ink-faint">
                computed with {m.metadata.tools.map((t) => t.tool).join(", ")}
              </div>
            )}
          </div>
        ))}

        {busy && <p className="text-[13px] text-ink-faint">Working it out…</p>}
        {error && <ErrorNote>{error}</ErrorNote>}
        <div ref={endRef} />
      </div>

      <div className="flex gap-2 border-t border-rule px-4 py-3">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(draft)}
          disabled={!hasDataset || busy}
          placeholder={hasDataset ? "Ask a question" : "Upload a table first"}
          className="flex-1 rounded border border-rule-strong px-2.5 py-1.5 text-[14px] placeholder:text-ink-faint disabled:bg-paper"
        />
        <Button onClick={() => send(draft)} disabled={!hasDataset || busy || !draft.trim()}>
          Ask
        </Button>
      </div>
    </section>
  );
}
