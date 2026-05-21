# Reading Vault

Personal knowledge vault for AI, math, and CS reading. Papers, blog posts,
and threads flow in from Readwise Reader via a Cloudflare Worker, get
triaged by Claude into structured notes, and surface via topic-based
dashboards.

## Architecture

```
Readwise Reader
     │ webhook (reader.any_document.created)
     ▼
Cloudflare Worker  ──verify secret──▶  GitHub Contents API
     │
     └─ writes:
        • 00_Inbox/raw/{YYYY-MM-DD}.md          (one bullet per item)
        • 00_Inbox/raw/payloads/{rw-id}.json    (full webhook payload)

You open Claude in this vault
     │
     ▼
SessionStart hook → git pull
     │
     ▼
/process-inbox     # triage today's items into deep notes
     │
     └─ writes:
        • 00_Inbox/{papers|blogs|threads|newsletters}/{date}.md
        • 30_Knowledge/{papers|blogs|threads}/{slug}.md
        • 30_Knowledge/topics/{topic}.md          (MOCs)
        • 30_Knowledge/authors/{author}.md
```

## Directory map

- `00_Inbox/raw/` — Worker writes here; raw bullets + payload JSON.
- `00_Inbox/{papers,blogs,threads,newsletters}/` — Claude routes into here during triage.
- `30_Knowledge/{papers,blogs,threads}/` — deep notes, one per item.
- `30_Knowledge/topics/` — topic MOCs (cross-cutting maps of content).
- `30_Knowledge/authors/` — per-author indices.
- `30_Knowledge/_MOCs/` — dashboards (recent, deep-read queue, orphans, …).
- `40_Archive/` — long-term storage for items no longer active.
- `50_Decisions/` — design decisions about the vault itself.
- `90_Meta/worker/` — Cloudflare Worker source.
- `90_Meta/scripts/` — utility scripts (snapshot, audit-topics, etc.).

## Setup (one-time)

See `90_Meta/worker/README.md` for full Worker deployment. Quick version:

1. Push this repo to a private GitHub repo named `reading-vault`.
2. Generate a Readwise access token: <https://readwise.io/access_token>
3. Generate a GitHub fine-grained PAT with Contents read/write scoped to this repo.
4. `cd 90_Meta/worker && npm install && wrangler login`
5. Set secrets:
   ```bash
   wrangler secret put READWISE_WEBHOOK_SECRET
   wrangler secret put GITHUB_TOKEN
   wrangler secret put READWISE_TOKEN
   ```
6. Edit `wrangler.toml` and set `GITHUB_OWNER` to your GitHub username.
7. `wrangler deploy` — note the Worker URL.
8. In Readwise, create a custom webhook pointing at the Worker URL. Subscribe to `reader.any_document.created`. Copy the `secret` Readwise generates and re-set the `READWISE_WEBHOOK_SECRET` to match.
9. Test by saving a paper to Readwise Reader. The Worker should create a commit in this repo within seconds.

## Usage

- Open Claude in the vault root: `cd /home/lukas/projects/reading-vault && claude`
- The `SessionStart` hook will `git pull` automatically.
- Run `/process-inbox` to triage what's accumulated.
- Ask "what should I read next?" or "what do we know about flow matching?" for dashboard-style queries.

See `CLAUDE.md` for the full operating spec.
