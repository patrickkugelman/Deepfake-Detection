const API_BASE = "http://localhost:8000";

const viewLogin = document.getElementById("view-login");
const viewAccount = document.getElementById("view-account");

// ── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  const apiKey = await getStorage("apiKey");
  if (apiKey) {
    showView("account");
    await loadUsage(apiKey);
    checkPendingResult();
  } else {
    showView("login");
  }
}

function showView(name) {
  viewLogin.classList.toggle("active", name === "login");
  viewAccount.classList.toggle("active", name === "account");
}

// ── Login ────────────────────────────────────────────────────────────────────

document.getElementById("saveKeyBtn").addEventListener("click", async () => {
  const key = document.getElementById("apiKeyInput").value.trim();
  if (!key.startsWith("sk_live_")) {
    showError("Key must start with sk_live_");
    return;
  }

  // Validate key against API
  try {
    const res = await fetch(`${API_BASE}/api/usage`, {
      headers: { "Authorization": `Bearer ${key}` },
    });
    if (res.status === 401) { showError("Invalid API key."); return; }
    if (!res.ok) { showError("Could not connect to DeepGuard server."); return; }
    const data = await res.json();
    await setStorage({ apiKey: key, username: data.username, plan: data.plan });
    showView("account");
    await loadUsage(key);
  } catch (e) {
    showError("Cannot reach server. Is DeepGuard running?");
  }
});

document.getElementById("apiKeyInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("saveKeyBtn").click();
});

// ── Logout ───────────────────────────────────────────────────────────────────

document.getElementById("logoutBtn").addEventListener("click", async () => {
  await chrome.storage.sync.remove(["apiKey", "username", "plan"]);
  await chrome.storage.local.remove(["lastResult", "analyzing"]);
  showView("login");
});

// ── Usage ────────────────────────────────────────────────────────────────────

async function loadUsage(apiKey) {
  try {
    const res = await fetch(`${API_BASE}/api/usage`, {
      headers: { "Authorization": `Bearer ${apiKey}` },
    });
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("usernameLabel").textContent = data.username || "";
    document.getElementById("planBadge").textContent = (data.plan || "free").toUpperCase();

    const used = data.used || 0;
    const limit = data.limit === -1 ? "∞" : data.limit;
    const pct = data.limit > 0 ? Math.min(100, (used / data.limit) * 100) : 0;

    document.getElementById("usageBar").style.width = `${pct}%`;
    if (pct > 80) document.getElementById("usageBar").style.background = "#f59e0b";
    if (pct >= 100) document.getElementById("usageBar").style.background = "#ef4444";

    document.getElementById("usageText").textContent = `${used} / ${limit} analyses this month`;
  } catch (e) {
    document.getElementById("usageText").textContent = "Could not load usage.";
  }
}

// ── Manual URL analysis ───────────────────────────────────────────────────────

document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const url = document.getElementById("urlInput").value.trim();
  if (!url || !url.startsWith("http")) {
    showError("Please enter a valid URL.");
    return;
  }
  const apiKey = await getStorage("apiKey");
  if (!apiKey) { showView("login"); return; }

  clearResult();
  setAnalyzing(true);

  const result = await sendToBackground("ANALYZE_URL", { url });
  setAnalyzing(false);

  if (result?.error) {
    showError(result.error === "quota_exceeded"
      ? "Monthly quota exceeded. Upgrade your plan."
      : `Error: ${result.error}`);
  } else if (result?.verdict) {
    showResult(result);
  }
});

document.getElementById("urlInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("analyzeBtn").click();
});

// ── Listen for background progress updates ────────────────────────────────────

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "PROGRESS") {
    // Could show a progress bar here — future enhancement
  }
});

// ── Check if background has a fresh result (from right-click) ────────────────

async function checkPendingResult() {
  const { lastResult, analyzing } = await chrome.storage.local.get(["lastResult", "analyzing"]);
  if (analyzing) {
    setAnalyzing(true);
    // Poll storage until result appears
    const poll = setInterval(async () => {
      const { lastResult: r, analyzing: a } = await chrome.storage.local.get(["lastResult", "analyzing"]);
      if (!a) {
        clearInterval(poll);
        setAnalyzing(false);
        if (r) showResult(r);
      }
    }, 1000);
  } else if (lastResult) {
    showResult(lastResult);
  }
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function setAnalyzing(on) {
  document.getElementById("analyzingRow").style.display = on ? "flex" : "none";
  document.getElementById("analyzeBtn").disabled = on;
}

function showResult(result) {
  const card = document.getElementById("resultCard");
  const verdict = (result.verdict || "INCONCLUSIVE").toUpperCase();
  const emojis = { FAKE: "🔴", REAL: "🟢", SUSPICIOUS: "🟡", INCONCLUSIVE: "⚪" };
  const classes = { FAKE: "fake", REAL: "real", SUSPICIOUS: "suspicious", INCONCLUSIVE: "" };

  card.className = `result-card ${classes[verdict] || ""}`;
  document.getElementById("verdictEmoji").textContent = emojis[verdict] || "⚪";
  document.getElementById("verdictLabel").textContent = verdict;

  const conf = result.confidence;
  const confPct = conf != null
    ? (conf <= 1 ? (conf * 100).toFixed(1) : parseFloat(conf).toFixed(1))
    : null;
  document.getElementById("confidenceLabel").textContent = confPct ? `${confPct}%` : "";
  document.getElementById("reasoningText").textContent = (result.reasoning || "").slice(0, 150);

  card.style.display = "block";
  document.getElementById("errorText").style.display = "none";
}

function showError(msg) {
  const el = document.getElementById("errorText");
  el.textContent = msg;
  el.style.display = "block";
}

function clearResult() {
  document.getElementById("resultCard").style.display = "none";
  document.getElementById("errorText").style.display = "none";
}

// ── Storage helpers ───────────────────────────────────────────────────────────

function getStorage(key) {
  return new Promise((resolve) => {
    chrome.storage.sync.get([key], (data) => resolve(data[key] || null));
  });
}

function setStorage(obj) {
  return new Promise((resolve) => chrome.storage.sync.set(obj, resolve));
}

function sendToBackground(type, data) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type, ...data }, resolve);
  });
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
