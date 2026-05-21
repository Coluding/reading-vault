/**
 * Readwise -> reading-vault ingest Worker.
 *
 * Receives Readwise custom webhooks, verifies the shared secret, writes two
 * files into the vault repo via the GitHub Contents API:
 *
 *   1. Appends a markdown bullet to `00_Inbox/raw/{YYYY-MM-DD}.md`
 *   2. Writes the full payload JSON to `00_Inbox/raw/payloads/{rw-id}.json`
 *
 * All categorisation, deep-note creation, and cross-linking is done by
 * Claude during `/process-inbox`. The Worker stays dumb on purpose.
 */

interface Env {
  // Secrets — set via `wrangler secret put`
  READWISE_WEBHOOK_SECRET: string;
  GITHUB_TOKEN: string;

  // Vars — set in wrangler.toml
  GITHUB_OWNER: string;
  GITHUB_REPO: string;
  GITHUB_BRANCH: string;
}

interface ReadwiseDocumentPayload {
  id: string;
  url: string;
  title: string;
  author: string | null;
  source: string | null;
  category: string | null;
  tags: Record<string, { name: string; type: string; created: number }> | null;
  site_name: string | null;
  word_count: number | null;
  reading_time: string | null;
  created_at: string;
  updated_at: string;
  published_date: string | null;
  summary: string | null;
  image_url: string | null;
  content: string | null;
  source_url: string | null;
  notes: string;
  parent_id: string | null;
  reading_progress: number;
  first_opened_at: string | null;
  last_opened_at: string | null;
  saved_at: string;
  last_moved_at: string;
  event_type: string;
  secret: string;
}

interface ReadwiseHighlightPayload {
  id: number;
  text: string;
  note: string;
  location: number | null;
  location_type: string;
  highlighted_at: string;
  url: string | null;
  color: string;
  updated: string;
  book_id: number;
  tags: string[];
  event_type: string;
  secret: string;
}

type ReadwisePayload = ReadwiseDocumentPayload | ReadwiseHighlightPayload;

function isDocument(p: ReadwisePayload): p is ReadwiseDocumentPayload {
  return "url" in p && "title" in p;
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function nowHHMM(): string {
  return new Date().toISOString().slice(11, 16);
}

function classify(p: ReadwiseDocumentPayload): "paper" | "thread" | "newsletter" | "blog" {
  const src = (p.source_url ?? p.url ?? "").toLowerCase();
  const cat = (p.category ?? "").toLowerCase();

  if (cat === "pdf") return "paper";
  if (
    src.includes("arxiv.org") ||
    src.includes("openreview.net") ||
    src.includes("proceedings.")
  ) {
    return "paper";
  }
  if (cat === "tweet" || src.includes("twitter.com") || src.includes("x.com") || src.includes("nitter.")) {
    return "thread";
  }
  if (cat === "email") {
    const site = (p.site_name ?? "").toLowerCase();
    if (
      site.includes("substack") ||
      site.includes("buttondown") ||
      src.includes("substack.com") ||
      src.includes("buttondown.email")
    ) {
      return "newsletter";
    }
    return "newsletter";
  }
  return "blog";
}

function formatBullet(p: ReadwiseDocumentPayload, type: string): string {
  const time = nowHHMM();
  const title = (p.title ?? "(no title)").replace(/\|/g, "\\|").replace(/\n/g, " ").trim();
  const author = (p.author ?? "").replace(/\|/g, "\\|").trim();
  const url = p.url ?? p.source_url ?? "";
  const tagNames = p.tags ? Object.values(p.tags).map((t) => t.name).join(", ") : "";
  const tagPart = tagNames ? ` | tags: ${tagNames}` : "";
  const authorPart = author ? ` — ${author}` : "";
  return `- [${type}] ${time} | "${title}"${authorPart} | ${url}${tagPart} | rw-id: ${p.id}`;
}

interface GhFileResponse {
  sha: string;
  content: string; // base64
}

async function ghGet(env: Env, path: string): Promise<GhFileResponse | null> {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/contents/${encodeURI(path)}?ref=${env.GITHUB_BRANCH}`;
  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      "User-Agent": "readwise-vault-ingest",
      Accept: "application/vnd.github+json",
    },
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`GH GET ${path} failed: ${res.status} ${await res.text()}`);
  return (await res.json()) as GhFileResponse;
}

async function ghPut(
  env: Env,
  path: string,
  contentUtf8: string,
  message: string,
  sha: string | null
): Promise<void> {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/contents/${encodeURI(path)}`;
  const body: Record<string, unknown> = {
    message,
    content: btoa(unescape(encodeURIComponent(contentUtf8))),
    branch: env.GITHUB_BRANCH,
  };
  if (sha) body.sha = sha;
  const res = await fetch(url, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      "User-Agent": "readwise-vault-ingest",
      Accept: "application/vnd.github+json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GH PUT ${path} failed: ${res.status} ${text}`);
  }
}

function decodeBase64Utf8(b64: string): string {
  const clean = b64.replace(/\n/g, "");
  const bin = atob(clean);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder("utf-8").decode(bytes);
}

/**
 * Read-modify-write with retry on SHA conflicts. GitHub's Contents API uses
 * optimistic locking: if the file changes between our GET and our PUT, the
 * PUT is rejected with 409/422. The `buildContent(current)` callback receives
 * the freshly-read content (or null if the file doesn't exist) and returns
 * the new content to PUT.
 */
async function ghPutWithRetry(
  env: Env,
  path: string,
  buildContent: (current: string | null) => string,
  message: string
): Promise<void> {
  const maxAttempts = 4;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const existing = await ghGet(env, path);
    const current = existing ? decodeBase64Utf8(existing.content) : null;
    const sha = existing ? existing.sha : null;
    const newContent = buildContent(current);
    try {
      await ghPut(env, path, newContent, message, sha);
      return;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // 409 / 422 typically mean stale sha — retry with fresh GET
      if (attempt < maxAttempts && /409|422|sha/i.test(msg)) {
        console.log(
          `ghPutWithRetry: SHA conflict on ${path}, attempt ${attempt}/${maxAttempts}, retrying`
        );
        await new Promise((r) => setTimeout(r, 200 * attempt));
        continue;
      }
      throw err;
    }
  }
}

async function appendBulletWithRetry(
  env: Env,
  path: string,
  bullet: string,
  header: string
): Promise<void> {
  await ghPutWithRetry(
    env,
    path,
    (current) => {
      if (current) {
        return current.endsWith("\n")
          ? current + bullet + "\n"
          : current + "\n" + bullet + "\n";
      }
      return header + "\n\n" + bullet + "\n";
    },
    `ingest: ${bullet.slice(0, 80)}`
  );
}

async function writeSidecarJson(env: Env, p: ReadwiseDocumentPayload): Promise<void> {
  const path = `00_Inbox/raw/payloads/${p.id}.json`;
  // Strip the per-webhook secret — it rotates and has no value in git history.
  const scrubbed = { ...p };
  delete (scrubbed as Partial<ReadwiseDocumentPayload>).secret;
  const content = JSON.stringify(scrubbed, null, 2) + "\n";
  await ghPutWithRetry(env, path, () => content, `ingest: payload ${p.id}`);
}

async function handleDocument(env: Env, p: ReadwiseDocumentPayload): Promise<Response> {
  const type = classify(p);
  const date = today();
  const inboxPath = `00_Inbox/raw/${date}.md`;
  const header = `# Raw inbox — ${date}\n\n_Worker-managed. Do not edit by hand; \`/process-inbox\` moves entries to \`00_Inbox/raw/processed/\`._`;

  // Write the sidecar first so the bullet's rw-id reference is always resolvable
  await writeSidecarJson(env, p);

  const bullet = formatBullet(p, type);
  await appendBulletWithRetry(env, inboxPath, bullet, header);

  return new Response(
    JSON.stringify({ ok: true, type, rw_id: p.id, inbox: inboxPath }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
}

async function handleHighlight(_env: Env, p: ReadwiseHighlightPayload): Promise<Response> {
  // Highlights are noise for this vault — log and 200.
  console.log(`highlight event ignored: id=${p.id} book_id=${p.book_id}`);
  return new Response(JSON.stringify({ ok: true, ignored: "highlight" }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function pingResponse(note: string): Response {
  return new Response(
    JSON.stringify({ ok: true, ping: note }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "GET") {
      return new Response("readwise-vault-ingest is alive\n", { status: 200 });
    }
    if (req.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    // Read as text first so we can distinguish "empty body" from "malformed JSON".
    // Readwise's "Test Endpoint" check sends a POST with an empty (or near-empty)
    // body before any secret has been generated — we must 200 those to pass the test.
    const bodyText = await req.text();
    if (!bodyText || bodyText.trim().length === 0) {
      console.log("ping: empty body");
      return pingResponse("empty body");
    }

    let payload: ReadwisePayload;
    try {
      payload = JSON.parse(bodyText) as ReadwisePayload;
    } catch {
      console.log(`ping: non-JSON body (first 200 chars): ${bodyText.slice(0, 200)}`);
      return pingResponse("non-JSON body acknowledged");
    }

    // Payload with no secret field — also a test ping. Don't 403; that would
    // fail Readwise's connectivity check after the webhook is saved.
    if (!payload.secret) {
      console.log("ping: payload has no secret field");
      return pingResponse("no secret field");
    }

    // Payload has a secret but it doesn't match — likely a misconfigured client
    // or an attacker. 403 here is the right signal.
    if (payload.secret !== env.READWISE_WEBHOOK_SECRET) {
      console.warn("rejected: secret mismatch");
      return new Response("Forbidden", { status: 403 });
    }

    try {
      if (isDocument(payload)) {
        return await handleDocument(env, payload);
      }
      return await handleHighlight(env, payload);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`ingest failed: ${msg}`);
      // Return 500 so Readwise will retry (per their delivery guarantees)
      return new Response(`Error: ${msg}`, { status: 500 });
    }
  },
};
