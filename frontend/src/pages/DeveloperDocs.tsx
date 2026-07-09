/**
 * Developer Portal — API Docs page
 * Interactive documentation with copy-ready code snippets.
 */
import { Link } from "react-router-dom";
import { API_URL } from "@/api";

function CodeBlock({ children, lang = "bash" }: { children: string; lang?: string }) {
  return (
    <div className="relative my-3">
      <div className="absolute top-2 right-2 text-xs bg-gray-700 text-white px-2 py-0.5 rounded z-10">{lang}</div>
      <pre className="bg-gray-900 text-gray-100 p-4 pt-8 rounded-lg overflow-x-auto text-sm">
        <code>{children}</code>
      </pre>
      <button
        onClick={() => navigator.clipboard.writeText(children)}
        className="absolute bottom-2 right-2 text-xs bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded"
      >📋 Copy</button>
    </div>
  );
}

function Endpoint({ method, path, desc }: { method: string; path: string; desc: string }) {
  const color = method === "GET" ? "bg-green-600"
    : method === "POST" ? "bg-blue-600"
    : method === "DELETE" ? "bg-red-600" : "bg-gray-600";
  return (
    <div className="flex items-start gap-3 py-2 border-b last:border-b-0">
      <span className={`${color} text-white text-xs font-bold px-2 py-1 rounded w-16 text-center`}>{method}</span>
      <code className="font-mono text-sm flex-shrink-0">{path}</code>
      <span className="text-gray-600 text-sm">{desc}</span>
    </div>
  );
}

export default function DeveloperDocs() {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">DeepGuard Pro API Documentation</h1>
        <Link to="/developer" className="text-sm text-indigo-600 hover:underline">← Back to Dashboard</Link>
      </div>

      <section>
        <h2 className="text-xl font-semibold mb-3">Authentication</h2>
        <p className="text-gray-700 mb-3">
          All requests require an API key in the <code className="bg-gray-100 px-1 rounded">X-API-Key</code> header.
          Generate one from the Developer Dashboard.
        </p>
        <CodeBlock>{`curl ${API_URL}/api/usage \\
  -H "X-API-Key: sk_live_YOUR_KEY"`}</CodeBlock>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Analyze a file (async)</h2>
        <p className="text-gray-700 mb-3">
          Upload a file and get a job ID back. Poll <code>/jobs/{"{id}"}</code> or register a webhook for the result.
        </p>
        <CodeBlock>{`curl -X POST ${API_URL}/chat_analyze \\
  -H "X-API-Key: sk_live_YOUR_KEY" \\
  -F "file=@photo.jpg" \\
  -F "query=Is this photo authentic?"`}</CodeBlock>
        <CodeBlock lang="python">{`import requests

r = requests.post(
    "${API_URL}/chat_analyze",
    headers={"X-API-Key": "sk_live_YOUR_KEY"},
    files={"file": open("photo.jpg", "rb")},
    data={"query": "Is this photo authentic?"},
)
print(r.json())  # {"job_id": "..."}`}</CodeBlock>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Analyze a URL</h2>
        <CodeBlock>{`curl -X POST ${API_URL}/analyze_url \\
  -H "X-API-Key: sk_live_YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"url": "https://example.com/video.mp4"}'`}</CodeBlock>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Check job status</h2>
        <CodeBlock>{`curl ${API_URL}/jobs/JOB_ID \\
  -H "X-API-Key: sk_live_YOUR_KEY"`}</CodeBlock>
        <p className="text-gray-600 text-sm">
          Returns <code>{"{ status, verdict, confidence, reasoning, details }"}</code> once completed.
        </p>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Webhooks (push delivery)</h2>
        <p className="text-gray-700 mb-3">
          Register a URL to receive analysis results automatically.
          Payloads are signed with HMAC-SHA256 — verify using the <code>X-DeepGuard-Signature</code> header.
        </p>
        <CodeBlock>{`curl -X POST ${API_URL}/webhooks/create \\
  -H "X-API-Key: sk_live_YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://your-server.com/hook",
    "events": ["analysis.completed", "analysis.failed"]
  }'`}</CodeBlock>
        <p className="text-gray-700 font-semibold mt-4 mb-2">Verify the signature (Python):</p>
        <CodeBlock lang="python">{`import hmac, hashlib

def verify(body_bytes: bytes, signature_header: str, secret: str) -> bool:
    mac = hmac.new(secret.encode(), body_bytes, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature_header)`}</CodeBlock>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Endpoint Reference</h2>
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow p-4">
          <Endpoint method="POST"   path="/chat_analyze" desc="Submit file for async analysis" />
          <Endpoint method="POST"   path="/analyze_url" desc="Analyze from URL" />
          <Endpoint method="GET"    path="/jobs/{id}" desc="Poll job status & retrieve verdict" />
          <Endpoint method="POST"   path="/api/keys" desc="Create new API key" />
          <Endpoint method="GET"    path="/api/keys" desc="List your API keys" />
          <Endpoint method="DELETE" path="/api/keys" desc="Revoke an API key" />
          <Endpoint method="GET"    path="/api/usage" desc="Current plan & quota" />
          <Endpoint method="GET"    path="/usage/dashboard" desc="Detailed usage breakdown" />
          <Endpoint method="POST"   path="/webhooks/create" desc="Register a webhook endpoint" />
          <Endpoint method="GET"    path="/webhooks/list" desc="List webhooks" />
          <Endpoint method="DELETE" path="/webhooks/{id}" desc="Delete a webhook" />
          <Endpoint method="POST"   path="/webhooks/{id}/test" desc="Send a test payload" />
          <Endpoint method="POST"   path="/generate-forensic-report" desc="Generate court-grade PDF (Pro+)" />
          <Endpoint method="GET"    path="/verify/{hash}" desc="Public report verification" />
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-3">Rate Limits</h2>
        <ul className="list-disc pl-5 text-gray-700 space-y-1">
          <li><b>Authenticated:</b> 60 requests / minute per API key</li>
          <li><b>Anonymous:</b> 10 requests / minute per IP</li>
          <li>Responses include <code>X-RateLimit-Limit</code>, <code>X-RateLimit-Remaining</code>, <code>X-RateLimit-Reset</code> headers</li>
          <li>Exceeding returns <b>HTTP 429</b> with <code>Retry-After</code></li>
        </ul>
      </section>
    </div>
  );
}
