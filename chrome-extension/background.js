const API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 3000;
const MAX_POLL_ATTEMPTS = 30; // 90 seconds max

// ── Context menu setup ──────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "deepguard-analyze",
    title: "🔍 Verify with DeepGuard",
    contexts: ["image", "video", "link"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "deepguard-analyze") return;

  const url = info.srcUrl || info.linkUrl || info.pageUrl;
  if (!url || url.startsWith("data:") || url.startsWith("blob:")) {
    showNotification("error", "Cannot analyze this type of media (data URI or blob URL).");
    return;
  }

  const apiKey = await getApiKey();
  if (!apiKey) {
    showNotification("error", "Please log in to DeepGuard first. Click the extension icon.");
    chrome.action.openPopup();
    return;
  }

  showNotification("analyzing", `Analyzing image...`);
  await analyzeUrl(url, apiKey);
});

// ── Message listener (from popup or content script) ─────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "ANALYZE_URL") {
    (async () => {
      const apiKey = await getApiKey();
      if (!apiKey) {
        sendResponse({ error: "Not authenticated" });
        return;
      }
      const result = await analyzeUrl(msg.url, apiKey);
      sendResponse(result);
    })();
    return true; // keep channel open for async response
  }

  if (msg.type === "GET_STATUS") {
    chrome.storage.local.get(["lastResult", "analyzing"], (data) => {
      sendResponse(data);
    });
    return true;
  }
});

// ── Core analysis logic ──────────────────────────────────────────────────────

async function analyzeUrl(mediaUrl, apiKey) {
  chrome.storage.local.set({ analyzing: true, lastResult: null });

  try {
    // Submit analysis job
    const response = await fetch(`${API_BASE}/analyze_url`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ url: mediaUrl }),
    });

    if (response.status === 401) {
      chrome.storage.local.set({ analyzing: false });
      showNotification("error", "Invalid API key. Please log in again.");
      return { error: "unauthorized" };
    }

    if (response.status === 429) {
      const data = await response.json();
      chrome.storage.local.set({ analyzing: false });
      showNotification("quota", data.detail?.message || "Monthly quota exceeded. Upgrade your plan.");
      return { error: "quota_exceeded" };
    }

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    const job = await response.json();

    // If already cached and completed instantly
    if (job.status === "completed" && job.verdict) {
      const result = {
        verdict: job.verdict,
        confidence: job.confidence,
        reasoning: job.reasoning,
        cached: true,
        url: mediaUrl,
      };
      chrome.storage.local.set({ analyzing: false, lastResult: result });
      showVerdictNotification(result);
      return result;
    }

    // Poll for result
    return await pollJob(job.job_id, apiKey, mediaUrl);

  } catch (err) {
    chrome.storage.local.set({ analyzing: false });
    showNotification("error", `Analysis failed: ${err.message}`);
    return { error: err.message };
  }
}

async function pollJob(jobId, apiKey, mediaUrl) {
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
    await sleep(POLL_INTERVAL_MS);

    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`, {
        headers: { "Authorization": `Bearer ${apiKey}` },
      });

      if (!res.ok) continue;
      const data = await res.json();

      if (data.status === "completed") {
        const result = {
          verdict: data.final_verdict || data.verdict || "INCONCLUSIVE",
          confidence: data.confidence,
          reasoning: data.reasoning || data.message,
          job_id: jobId,
          url: mediaUrl,
        };
        chrome.storage.local.set({ analyzing: false, lastResult: result });
        showVerdictNotification(result);
        return result;
      }

      if (data.status === "failed") {
        chrome.storage.local.set({ analyzing: false });
        showNotification("error", "Analysis failed on the server. Please try again.");
        return { error: "analysis_failed" };
      }

      // Update popup with progress if open
      chrome.runtime.sendMessage({ type: "PROGRESS", progress: data.progress || 0 }).catch(() => {});

    } catch (e) {
      // Network error — keep polling
    }
  }

  chrome.storage.local.set({ analyzing: false });
  showNotification("error", "Analysis timed out. Please try again.");
  return { error: "timeout" };
}

// ── Notifications ────────────────────────────────────────────────────────────

function showVerdictNotification(result) {
  const verdict = (result.verdict || "INCONCLUSIVE").toUpperCase();
  const icons = { FAKE: "🔴", REAL: "🟢", SUSPICIOUS: "🟡", INCONCLUSIVE: "⚪" };
  const icon = icons[verdict] || "⚪";
  const conf = result.confidence
    ? (result.confidence <= 1 ? (result.confidence * 100).toFixed(1) : result.confidence.toFixed(1))
    : "—";

  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon128.png",
    title: `${icon} DeepGuard: ${verdict}`,
    message: `Confidence: ${conf}% — ${(result.reasoning || "").slice(0, 100)}`,
    priority: verdict === "FAKE" ? 2 : 1,
  });
}

function showNotification(type, message) {
  const titles = {
    analyzing: "🔍 DeepGuard — Analyzing...",
    error: "❌ DeepGuard — Error",
    quota: "⚠️ DeepGuard — Quota Exceeded",
  };
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon128.png",
    title: titles[type] || "DeepGuard",
    message,
    priority: type === "error" ? 2 : 0,
  });
}

// ── Storage helpers ──────────────────────────────────────────────────────────

function getApiKey() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["apiKey"], (data) => resolve(data.apiKey || null));
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
