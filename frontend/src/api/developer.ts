/**
 * Developer Portal API client — DeepGuard Pro SaaS (Module 8)
 *
 * Exposes:
 *  - API key management (/api/keys)
 *  - Usage / plan / dashboard
 *  - Outbound webhooks (/webhooks/*)
 *  - Forensic reports (/generate-forensic-report, /verify/:hash)
 *  - Billing (Stripe — endpoints gracefully degrade if module 4 is not wired yet)
 */
import { api } from "./index";

// ========= TYPES =========
export interface ApiKey {
  prefix: string;
  name?: string;
  is_active: boolean;
  created_at?: string | null;
  last_used?: string | null;
}

export interface CreatedApiKey extends ApiKey {
  key: string; // full key, shown once
  warning?: string;
}

export interface UsageSummary {
  username: string;
  plan: string;
  used: number;
  limit: number;
  remaining: number;
  reset: string;
  plans: Record<string, { limit: number; price_usd: number }>;
}

export interface DailyPoint {
  date: string;
  requests: number;
}

export interface UsageDashboard {
  plan: string;
  monthly_limit: number;
  used_this_month: number;
  verdict_breakdown?: Record<string, number>;
  file_type_breakdown?: Record<string, number>;
  per_key?: Array<{ prefix: string; name?: string; requests: number }>;
  daily_chart: DailyPoint[];
}

export interface Webhook {
  webhook_id: string;
  url: string;
  events: string[];
  is_active: boolean;
  created_at?: string | null;
  last_triggered_at?: string | null;
  failure_count: number;
  success_count: number;
}

export interface WebhookCreated extends Webhook {
  secret: string; // shown once
  warning: string;
}

// ========= API KEYS =========
export const listApiKeys = async (): Promise<ApiKey[]> => {
  const r = await api.get("/api/keys");
  return r.data.keys || [];
};

export const createApiKey = async (name: string): Promise<CreatedApiKey> => {
  const r = await api.post("/api/keys", { name });
  return r.data;
};

export const revokeApiKey = async (prefix: string): Promise<void> => {
  await api.delete("/api/keys", { data: { prefix } });
};

export const getKeyUsage = async (prefix: string) => {
  const r = await api.get(`/api/keys/${prefix}/usage`);
  return r.data;
};

// ========= USAGE =========
export const getUsageSummary = async (): Promise<UsageSummary> => {
  const r = await api.get("/api/usage");
  return r.data;
};

export const getUsageDashboard = async (): Promise<UsageDashboard> => {
  const r = await api.get("/usage/dashboard");
  return r.data;
};

// ========= WEBHOOKS =========
export const listWebhooks = async (): Promise<Webhook[]> => {
  const r = await api.get("/webhooks/list");
  return r.data.webhooks || [];
};

export const createWebhook = async (url: string, events: string[]): Promise<WebhookCreated> => {
  const r = await api.post("/webhooks/create", { url, events });
  return r.data;
};

export const deleteWebhook = async (webhookId: string): Promise<void> => {
  await api.delete(`/webhooks/${webhookId}`);
};

export const testWebhook = async (webhookId: string) => {
  const r = await api.post(`/webhooks/${webhookId}/test`);
  return r.data;
};

// ========= FORENSIC REPORTS =========
export const downloadForensicReport = async (jobId: string): Promise<Blob> => {
  const r = await api.post(
    "/generate-forensic-report",
    { job_id: jobId },
    { responseType: "blob" }
  );
  return r.data;
};

// ========= BILLING (Stripe — Module 4, graceful if missing) =========
export const createCheckoutSession = async (plan: "pro" | "business") => {
  try {
    const r = await api.post("/billing/checkout", { plan });
    return r.data;
  } catch (e) {
    return { error: "billing_not_configured", message: "Stripe is not yet configured." };
  }
};

export const getBillingStatus = async () => {
  try {
    const r = await api.get("/billing/status");
    return r.data;
  } catch {
    return null;
  }
};
