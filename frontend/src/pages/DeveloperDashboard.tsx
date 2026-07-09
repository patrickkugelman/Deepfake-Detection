/**
 * Developer Portal — Dashboard page
 * Plan card + API keys table + usage chart (30-day)
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  listApiKeys,
  createApiKey,
  revokeApiKey,
  getUsageSummary,
  getUsageDashboard,
  type ApiKey,
  type UsageSummary,
  type UsageDashboard,
} from "@/api/developer";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

export default function DeveloperDashboard() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [dash, setDash] = useState<UsageDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [newKeyName, setNewKeyName] = useState("");
  const [justCreated, setJustCreated] = useState<{ key: string; name?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const [k, u, d] = await Promise.all([
        listApiKeys(),
        getUsageSummary(),
        getUsageDashboard().catch(() => null as any),
      ]);
      setKeys(k);
      setUsage(u);
      setDash(d);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Failed to load developer portal");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async () => {
    if (!newKeyName.trim()) return;
    try {
      const created = await createApiKey(newKeyName.trim());
      setJustCreated({ key: created.key, name: created.name });
      setNewKeyName("");
      await load();
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Failed to create key");
    }
  };

  const handleRevoke = async (prefix: string) => {
    if (!confirm(`Revoke key ${prefix}? Existing integrations will break.`)) return;
    await revokeApiKey(prefix);
    await load();
  };

  if (loading) return <div className="p-10 text-center">Loading developer portal...</div>;

  const pct = usage && usage.limit > 0 ? Math.min(100, (usage.used / usage.limit) * 100) : 0;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Developer Portal</h1>
        <nav className="flex gap-3 text-sm">
          <Link to="/developer/docs" className="text-blue-600 hover:underline">📚 Docs</Link>
          <Link to="/developer/webhooks" className="text-blue-600 hover:underline">🪝 Webhooks</Link>
          <Link to="/billing" className="text-blue-600 hover:underline">💳 Billing</Link>
        </nav>
      </div>

      {error && <div className="bg-red-100 border border-red-300 text-red-700 p-3 rounded">{error}</div>}

      {/* Plan card */}
      {usage && (
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 rounded-xl shadow-lg">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-sm opacity-80 uppercase tracking-wide">Current Plan</div>
              <div className="text-4xl font-bold mt-1">{usage.plan.toUpperCase()}</div>
            </div>
            <Link to="/billing" className="bg-white text-indigo-700 px-4 py-2 rounded-lg font-semibold hover:bg-gray-100">
              {usage.plan === "free" ? "Upgrade →" : "Manage"}
            </Link>
          </div>
          <div className="mt-6">
            <div className="flex justify-between text-sm mb-2">
              <span>{usage.used.toLocaleString()} / {usage.limit === -1 ? "∞" : usage.limit.toLocaleString()} requests this month</span>
              <span>{usage.reset}</span>
            </div>
            <div className="w-full bg-white/20 rounded-full h-3">
              <div className="bg-white rounded-full h-3 transition-all" style={{ width: `${pct}%` }} />
            </div>
          </div>
        </div>
      )}

      {/* 30-day chart */}
      {dash && dash.daily_chart && dash.daily_chart.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Usage — Last 30 days</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={dash.daily_chart}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
              <Tooltip />
              <Line type="monotone" dataKey="requests" stroke="#6366f1" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* API Keys */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">API Keys</h2>
        </div>

        <div className="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="Name this key (e.g. production, staging)"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            className="flex-1 border rounded px-3 py-2 dark:bg-gray-700 dark:border-gray-600"
          />
          <button
            onClick={handleCreate}
            disabled={!newKeyName.trim()}
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            + New Key
          </button>
        </div>

        {justCreated && (
          <div className="bg-yellow-50 border border-yellow-300 p-4 rounded mb-4">
            <div className="font-semibold mb-2">⚠️ Copy your key now — it won't be shown again.</div>
            <div className="font-mono text-sm bg-white p-3 rounded border select-all break-all">{justCreated.key}</div>
            <button
              onClick={() => { navigator.clipboard.writeText(justCreated.key); }}
              className="mt-2 text-sm text-indigo-600 hover:underline"
            >📋 Copy to clipboard</button>
          </div>
        )}

        {keys.length === 0 ? (
          <p className="text-gray-500 text-sm">No API keys yet. Create one above to get started.</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left">
                <th className="py-2">Name</th>
                <th>Prefix</th>
                <th>Created</th>
                <th>Last used</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k.prefix} className="border-b">
                  <td className="py-2">{k.name || "—"}</td>
                  <td className="font-mono">{k.prefix}…</td>
                  <td>{k.created_at ? new Date(k.created_at).toLocaleDateString() : "—"}</td>
                  <td>{k.last_used ? new Date(k.last_used).toLocaleDateString() : "never"}</td>
                  <td>
                    <span className={k.is_active
                      ? "text-green-600 font-medium"
                      : "text-gray-400"}>
                      {k.is_active ? "● active" : "● revoked"}
                    </span>
                  </td>
                  <td className="text-right">
                    {k.is_active && (
                      <button onClick={() => handleRevoke(k.prefix)}
                        className="text-red-600 hover:underline text-xs">Revoke</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
