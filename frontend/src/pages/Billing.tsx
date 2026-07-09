/**
 * Billing page — plan cards + Stripe checkout redirect
 * Falls back gracefully when Module 4 (Stripe) isn't wired yet.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getUsageSummary, createCheckoutSession, type UsageSummary } from "@/api/developer";

const PLANS = [
  {
    id: "free",
    name: "Free",
    price: 0,
    requests: 100,
    features: ["100 analyses / month", "Basic verdicts", "Email support"],
    cta: "Current free tier",
  },
  {
    id: "pro",
    name: "Pro",
    price: 49,
    requests: 2000,
    highlight: true,
    features: [
      "2,000 analyses / month",
      "Forensic PDF reports",
      "Webhooks",
      "Priority support",
      "Full agent details",
    ],
    cta: "Upgrade to Pro",
  },
  {
    id: "business",
    name: "Business",
    price: 149,
    requests: 10000,
    features: [
      "10,000 analyses / month",
      "Everything in Pro",
      "SLA 99.9% uptime",
      "Dedicated Slack channel",
      "Custom integrations",
    ],
    cta: "Go Business",
  },
];

export default function Billing() {
  const [usage, setUsage] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [redirecting, setRedirecting] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getUsageSummary().then(setUsage).finally(() => setLoading(false));
  }, []);

  const handleUpgrade = async (planId: "pro" | "business") => {
    setRedirecting(planId);
    setErr(null);
    const session = await createCheckoutSession(planId);
    if (session.error === "billing_not_configured" || !session.checkout_url) {
      setErr("Stripe billing is not yet configured on this deployment. Please contact support.");
      setRedirecting(null);
      return;
    }
    window.location.href = session.checkout_url;
  };

  if (loading) return <div className="p-10 text-center">Loading billing...</div>;

  const currentPlan = (usage?.plan || "free").toLowerCase();

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Billing & Plans</h1>
        <Link to="/developer" className="text-sm text-indigo-600 hover:underline">← Back to Dashboard</Link>
      </div>

      {usage && (
        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg flex justify-between">
          <div>
            <div className="text-sm text-gray-500">You are on</div>
            <div className="text-2xl font-bold">{currentPlan.toUpperCase()}</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">This month</div>
            <div className="text-2xl font-bold">
              {usage.used.toLocaleString()} / {usage.limit === -1 ? "∞" : usage.limit.toLocaleString()}
            </div>
          </div>
        </div>
      )}

      {err && <div className="bg-red-100 border border-red-300 text-red-700 p-3 rounded">{err}</div>}

      <div className="grid md:grid-cols-3 gap-6">
        {PLANS.map(p => {
          const isCurrent = p.id === currentPlan;
          return (
            <div
              key={p.id}
              className={`rounded-xl p-6 shadow-lg border-2 ${
                p.highlight
                  ? "border-indigo-500 bg-gradient-to-b from-indigo-50 to-white dark:from-indigo-900/30 dark:to-gray-800"
                  : "border-gray-200 bg-white dark:bg-gray-800 dark:border-gray-700"
              }`}
            >
              {p.highlight && (
                <div className="text-xs font-bold text-indigo-600 uppercase tracking-wide mb-2">Most popular</div>
              )}
              <h3 className="text-2xl font-bold">{p.name}</h3>
              <div className="mt-2 mb-4">
                <span className="text-4xl font-extrabold">${p.price}</span>
                <span className="text-gray-500">/mo</span>
              </div>
              <ul className="space-y-2 text-sm">
                {p.features.map(f => (
                  <li key={f} className="flex gap-2">
                    <span className="text-green-600">✓</span>
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
              <button
                disabled={isCurrent || redirecting === p.id || p.id === "free"}
                onClick={() => handleUpgrade(p.id as "pro" | "business")}
                className={`mt-6 w-full py-3 rounded-lg font-semibold ${
                  isCurrent
                    ? "bg-gray-200 text-gray-500 cursor-not-allowed dark:bg-gray-700"
                    : "bg-indigo-600 text-white hover:bg-indigo-700"
                } disabled:opacity-60`}
              >
                {isCurrent ? "Current plan" : redirecting === p.id ? "Redirecting..." : p.cta}
              </button>
            </div>
          );
        })}
      </div>

      <div className="text-xs text-gray-500 text-center">
        Prices in USD. Cancel anytime from the Stripe customer portal. Questions?
        Contact <a href="mailto:support@deepguard.io" className="text-indigo-600 hover:underline">support@deepguard.io</a>.
      </div>
    </div>
  );
}
