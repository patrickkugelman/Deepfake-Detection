/**
 * Developer Portal — Webhooks management
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  listWebhooks, createWebhook, deleteWebhook, testWebhook,
  type Webhook,
} from "@/api/developer";

const SUPPORTED_EVENTS = [
  { id: "analysis.completed", label: "Analysis completed", desc: "Fired when a job finishes successfully." },
  { id: "analysis.failed",    label: "Analysis failed",    desc: "Fired when a job errors out." },
  { id: "quota.exceeded",     label: "Quota exceeded",     desc: "Fired when monthly quota is reached." },
];

export default function DeveloperWebhooks() {
  const [hooks, setHooks] = useState<Webhook[]>([]);
  const [loading, setLoading] = useState(true);
  const [url, setUrl] = useState("");
  const [events, setEvents] = useState<string[]>(["analysis.completed"]);
  const [justCreated, setJustCreated] = useState<{ secret: string; url: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      setHooks(await listWebhooks());
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Failed to load webhooks");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const toggleEvent = (id: string) => {
    setEvents(events.includes(id) ? events.filter(e => e !== id) : [...events, id]);
  };

  const handleCreate = async () => {
    setError(null);
    if (!url.match(/^https?:\/\//)) {
      setError("URL must start with http:// or https://"); return;
    }
    if (events.length === 0) { setError("Select at least one event"); return; }
    try {
      const created = await createWebhook(url, events);
      setJustCreated({ secret: created.secret, url: created.url });
      setUrl("");
      setEvents(["analysis.completed"]);
      await load();
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Failed to create webhook");
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this webhook? It will stop receiving events.")) return;
    await deleteWebhook(id);
    await load();
  };

  const handleTest = async (id: string) => {
    try {
      await testWebhook(id);
      alert("Test payload queued. Check your endpoint in a few seconds.");
    } catch (e: any) {
      alert("Test failed: " + (e?.response?.data?.detail || e.message));
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Webhooks</h1>
        <Link to="/developer" className="text-sm text-indigo-600 hover:underline">← Back to Dashboard</Link>
      </div>

      <p className="text-gray-600">
        Webhooks push analysis results to your server automatically — no polling needed.
        Payloads are signed with HMAC-SHA256 using the secret shown once at creation.
      </p>

      {error && <div className="bg-red-100 border border-red-300 text-red-700 p-3 rounded">{error}</div>}

      {/* Create form */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Create new webhook</h2>
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium">Endpoint URL</label>
            <input
              type="url"
              placeholder="https://your-server.com/hook"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="mt-1 w-full border rounded px-3 py-2 dark:bg-gray-700 dark:border-gray-600"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Events</label>
            <div className="mt-2 space-y-2">
              {SUPPORTED_EVENTS.map(ev => (
                <label key={ev.id} className="flex items-start gap-2 p-2 rounded hover:bg-gray-50 dark:hover:bg-gray-700">
                  <input
                    type="checkbox"
                    checked={events.includes(ev.id)}
                    onChange={() => toggleEvent(ev.id)}
                    className="mt-1"
                  />
                  <div>
                    <code className="font-mono text-sm">{ev.id}</code>
                    <div className="text-xs text-gray-500">{ev.desc}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
          <button
            onClick={handleCreate}
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700"
          >Create webhook</button>
        </div>

        {justCreated && (
          <div className="mt-6 bg-yellow-50 border border-yellow-300 p-4 rounded">
            <div className="font-semibold mb-2">⚠️ Save your signing secret — it won't be shown again.</div>
            <div className="text-xs text-gray-600 mb-2">For: <code>{justCreated.url}</code></div>
            <div className="font-mono text-sm bg-white p-3 rounded border select-all break-all">{justCreated.secret}</div>
            <button
              onClick={() => { navigator.clipboard.writeText(justCreated.secret); }}
              className="mt-2 text-sm text-indigo-600 hover:underline"
            >📋 Copy to clipboard</button>
          </div>
        )}
      </div>

      {/* List */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Registered webhooks</h2>
        {loading ? <p>Loading...</p> :
          hooks.length === 0 ? <p className="text-gray-500 text-sm">No webhooks yet.</p> :
          <div className="space-y-3">
            {hooks.map(h => (
              <div key={h.webhook_id} className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-sm truncate">{h.url}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {h.events.map(ev => (
                        <span key={ev} className="inline-block bg-gray-100 dark:bg-gray-700 rounded px-2 py-0.5 mr-1">{ev}</span>
                      ))}
                    </div>
                    <div className="text-xs mt-2 flex gap-4">
                      <span className={h.is_active ? "text-green-600" : "text-red-500"}>
                        ● {h.is_active ? "active" : "disabled"}
                      </span>
                      <span className="text-gray-500">✓ {h.success_count} delivered</span>
                      <span className="text-gray-500">⚠ {h.failure_count} failed</span>
                    </div>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <button onClick={() => handleTest(h.webhook_id)}
                      className="text-xs px-3 py-1 border rounded hover:bg-gray-50 dark:hover:bg-gray-700">Test</button>
                    <button onClick={() => handleDelete(h.webhook_id)}
                      className="text-xs text-red-600 hover:underline">Delete</button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        }
      </div>
    </div>
  );
}
