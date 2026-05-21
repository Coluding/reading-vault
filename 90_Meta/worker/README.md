# readwise-vault-ingest

Cloudflare Worker that ingests Readwise Reader webhooks into the
`reading-vault` GitHub repo.

## What it does

For each `reader.any_document.created` webhook, the Worker:

1. Verifies the `secret` field matches `READWISE_WEBHOOK_SECRET`.
2. Classifies the document (`paper | thread | newsletter | blog`) based on
   the payload's `category` and `source_url`.
3. Writes the full webhook payload (minus the rotating secret) to
   `00_Inbox/raw/payloads/{rw-id}.json` in the vault repo.
4. Appends a markdown bullet to `00_Inbox/raw/{YYYY-MM-DD}.md`.

It does **not** create deep notes, route to category inboxes, or fetch
full content — that's all Claude's job during `/process-inbox`. The
Worker stays dumb so heuristics can evolve without redeploys.

## Setup

### 1. Cloudflare tooling

```bash
cd 90_Meta/worker
npm install
npx wrangler login
```

### 2. GitHub PAT (fine-grained)

Generate at <https://github.com/settings/personal-access-tokens/new>:

- **Resource owner**: you
- **Repository access**: only `reading-vault`
- **Permissions** → Repository:
  - **Contents**: Read and write
  - **Metadata**: Read (automatic)
- **Expiration**: 1 year (renew annually)

Copy the token.

### 3. Readwise access token

Generate at <https://readwise.io/access_token>. Copy it — this is used by
Claude during triage, **not** by the Worker. Store it where Claude can
find it in the vault (e.g. `~/.config/readwise/token` or `.env` on your
machine — not committed).

### 4. Worker config

Edit `wrangler.toml` and replace `GITHUB_OWNER` with your GitHub username.

### 5. Webhook secret

You'll set this in two places (Readwise generates the secret automatically
when you create a custom webhook). For now, pick any 32-char string —
you'll replace it once Readwise gives you the real one.

### 6. Set secrets

```bash
npx wrangler secret put GITHUB_TOKEN
# paste the GitHub PAT

npx wrangler secret put READWISE_WEBHOOK_SECRET
# paste any 32-char placeholder for now
```

### 7. Deploy

```bash
npx wrangler deploy
```

Wrangler will print the Worker URL, something like
`https://readwise-vault-ingest.{your-subdomain}.workers.dev`. Copy it.

### 8. Configure the Readwise webhook

1. Go to <https://readwise.io/dashboard/webhooks> (or Readwise → Settings → Webhooks).
2. Click **Create New Webhook**.
3. Fill in:
   - **Name**: `Reading vault ingest`
   - **URL**: the Worker URL from step 7
   - **Event Types**: check `reader.any_document.created`
4. Click **Test Endpoint** — should return 200 from the Worker (test payloads have a placeholder secret; expect a 403 if the placeholder doesn't match — see below for fix).
5. After saving, Readwise shows you the **secret** for this webhook. Copy it.

### 9. Replace the webhook secret with the real one

```bash
npx wrangler secret put READWISE_WEBHOOK_SECRET
# paste the secret Readwise generated
npx wrangler deploy   # not strictly needed; secrets update without redeploy
```

### 10. Test end-to-end

Save a paper to Readwise Reader (e.g. an arXiv link). Within a few
seconds you should see:

- A commit in the vault repo authored by your PAT identity
- `00_Inbox/raw/{today}.md` containing a new bullet
- `00_Inbox/raw/payloads/{rw-id}.json` with the full payload

If it doesn't work, run `npx wrangler tail` and trigger another save.

## Development

```bash
npm run dev      # local dev server (won't write to real GH unless you use real secrets)
npm run deploy   # deploy to production
npm run tail     # follow production logs
```

## Updating the classification rules

The classification logic lives in `src/index.ts` in the `classify()`
function. If you change it, also update `CLAUDE.md` Part 5 in the vault
root to keep the spec and the code in sync. Claude reads `CLAUDE.md` as
the source of truth — the Worker just needs to roughly agree.

## Cost

Worker is free for any reasonable reading volume (free tier covers
100k requests/day). GitHub Contents API is also free for this volume.
The only thing that costs money is your Readwise subscription, which
you already have.
