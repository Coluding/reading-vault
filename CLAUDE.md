# CLAUDE.md — Operating Instructions for the Reading Vault

You are Claude Code, working in a personal reading vault for AI, math, and
computer science material. Papers, blog posts, and Twitter/X threads arrive
here automatically via a Cloudflare Worker connected to Readwise Reader.
This file is your behavioural spec. Read it on every session.

---

## Part 1 — What this vault is

A second brain for everything the user reads in the AI / math / CS space.
Items flow in continuously from Readwise Reader (papers, blog posts,
threads, newsletters). The vault's job is to:

1. **Capture** every item without manual filing (the Worker handles this).
2. **Triage** items from the raw inbox into category-specific inboxes.
3. **Deepen** each item into a structured note with cross-links into topic
   MOCs and author indices.
4. **Surface** what the user should read next via dashboards (MOCs).

The vault is generative: every triage pass produces deeper notes, every
deep read produces fresh cross-links, and the dashboards re-rank what
matters. **You write to it constantly. Don't ask permission to file.**

---

## Part 2 — The three modes you operate in

### Mode A — Ingest (automatic, not you)

The Cloudflare Worker at `90_Meta/worker/` receives webhooks from Readwise
and writes raw entries to `00_Inbox/raw/{YYYY-MM-DD}.md` plus full payload
JSON to `00_Inbox/raw/payloads/{rw-id}.json`. You don't do this — it
happens whether or not Claude is running.

### Mode B — Triage (default when user opens vault)

The user runs `/process-inbox` or says "process today's inbox". You:

1. Read every unprocessed bullet from `00_Inbox/raw/*.md` (anything not in
   `00_Inbox/raw/processed/`).
2. For each bullet, read its sidecar JSON at `00_Inbox/raw/payloads/{rw-id}.json`.
3. **Classify** into `paper | blog | thread | newsletter | reject` using
   the rules in Part 5.
4. **Route**: append the bullet to the category inbox
   `00_Inbox/{type}/{YYYY-MM-DD}.md`.
5. **Deepen**: create `30_Knowledge/{type}/{slug}.md` with the frontmatter
   in Part 7 and a **substantial** body. These notes are the durable record — depth here saves you from re-reading the source later. Body templates differ by type (see Part 12).

   **Hard requirement for all types**: no fabricated quotes, numbers, or claims. Everything stated as fact must come from the fetched content (Readwise API for papers/blogs, webhook payload summary for threads/newsletters). Your synthesis must be labelled `[analyst's view]` so future-you can tell facts from interpretation. If a template section can't be filled because the source doesn't address it, write `_not addressed by the source_` instead of inventing.
6. For papers and blogs, **fetch full content** via the Readwise Document
   Export API using the rw-id. Use the full text to write the TL;DR and
   key claims. For threads and newsletters, the webhook payload's summary
   is sufficient.
7. **Update topic MOCs**: for each topic the note touches, add a
   `[[link]]` to `30_Knowledge/topics/{topic}.md` (create the MOC if
   missing).
8. **Update author indices**: same for `30_Knowledge/authors/{author}.md`.
9. **Move processed bullets** from `00_Inbox/raw/{date}.md` to
   `00_Inbox/raw/processed/{date}.md`. Keep the JSON sidecars in place —
   they're the durable record.

### Mode C — Synthesis (on-demand queries)

The user asks dashboard-style questions:

- "What should I read next?" → read `30_Knowledge/_MOCs/dashboard-deep-read.md` and recent notes; recommend with reasoning.
- "What do we know about {topic}?" → read `30_Knowledge/topics/{topic}.md` and the linked notes; synthesise.
- "What patterns are emerging?" → scan last 30 days of `30_Knowledge/**/*.md`; name themes.
- "What am I ignoring?" → list notes with `read_state: queued` older than 14 days, orphan notes with no inbound links, and stale topic MOCs.

Synthesis output must cite specific notes via `[[wikilinks]]`. Never
fabricate findings — if the vault doesn't know, say so.

### Mode D — Drafting (turning reading into writing)

The user wants to **write** blog posts, not just file what they read. This lives
in `50_Writing/` (see Part 14). Triggers:

- "Start a blog on {topic}" → create `50_Writing/{slug}/` from `50_Writing/_template/`, set the angle in `index.md`, then run a first research pass (Mode-D gather).
- "Gather research for the {slug} blog" → re-scan `30_Knowledge/**` for relevant notes (`grep` by topic/frontmatter), append findings to `research.md`, **each linked to its source note** via `[[wikilinks]]`.
- "Outline / draft the {slug} blog" → work `draft.md` from `research.md`; outline before prose.

The same **no-fabrication** rule applies: everything in `research.md` must trace
to a `30_Knowledge` note (which traces to a real source). If the piece needs
something the vault doesn't have, mark it `_needs note_` in `research.md` and add
it to the reading queue — don't write from memory. `draft.md` prose is the user's
voice; Claude scaffolds and gathers, and only ghost-writes sections when asked.

---

## Part 3 — Hard rules

1. **Never delete files.** Move to `40_Archive/` instead.
2. **Never fabricate paper findings, numbers, or quotes.** If you didn't fetch
   the content via the Readwise API or read it from a webhook payload, you
   don't know it. Write `_needs verification_` and stop.
3. **Always cite the rw-id** in deep notes via the `rw_id` frontmatter
   field, so future-you can re-fetch the source.
4. **Never invent new note types in `30_Knowledge`** (`papers | blogs | threads | newsletters`
   are the only categories there). Ask before extending. _(The `50_Writing/`
   workspace has its own draft types — `blog-draft | blog-research | blog-draft-body` —
   per Part 14; these are separate and do not count as Knowledge categories.)_
5. **Never edit `90_Meta/`** unless the user explicitly asks (that's
   Worker code and scripts).
6. **One commit per `/process-inbox` run.** The Worker writes commit-per-
   webhook; your triage pass batches into a single commit at the end.
7. **Reject category includes everything off-topic for AI/math/CS.** If a
   webhook payload is clearly off-topic (e.g. a Verge product email),
   mark `reject: off-topic` in the bullet and move it straight to
   processed/ without a deep note. Don't silently drop it — the audit
   trail matters.

---

## Part 4 — Reading discipline

**At session start the `SessionStart` hook injects two things into your context automatically:**

1. The result of `git pull` (fresh state from the GitHub remote).
2. The output of `90_Meta/scripts/inbox-status.sh` — a one-paragraph summary of unprocessed raw inbox items.

If the inbox-status line reports **non-zero unprocessed items**, **proactively surface this to the user as your first response** in the session — even before they ask anything. Format:

> "You have N unprocessed item(s) in the inbox ({breakdown by day}). Want me to run `/process-inbox` to triage?"

If the inbox is empty, do **not** mention it — silence is the right signal.

**Always read at session start:**

- `CLAUDE.md` (this file)
- `30_Knowledge/_MOCs/dashboard-recent.md` (so you know what landed lately)

**Read on demand:**

- Specific raw inbox files when triaging
- Specific sidecar payload JSONs for items being processed
- Specific topic MOCs when the user asks about a topic
- Specific deep notes when the user names them

**Never read proactively:**

- The full `00_Inbox/raw/processed/` archive
- The full `30_Knowledge/papers/` directory
- Old payload JSONs

Find notes by frontmatter, not by reading them:

```bash
grep -l "topic: flow-matching" 30_Knowledge/papers/*.md
grep -l "read_state: deep" 30_Knowledge/**/*.md
```

---

## Part 5 — Classification rules

Apply in order; first match wins.

| Signal | Classification |
|---|---|
| `source_url` matches `arxiv.org` / `openreview.net` / `proceedings.*` / `category == "pdf"` | **paper** |
| `category == "tweet"` OR `source_url` matches `twitter.com` / `x.com` / `nitter.*` | **thread** |
| `category == "email"` AND `site_name` is a newsletter (Substack, Buttondown, etc.) | **newsletter** |
| `category == "article"` AND domain is a known blog (lilianweng.github.io, distill.pub, ai.googleblog.com, openai.com/blog, anthropic.com/news, personal-domain blogs) | **blog** |
| `category == "email"` AND looks promotional (low signal sender, marketing copy in title) | **reject: off-topic** |
| Anything else | **blog** (default) — if it turns out to be off-topic, reject in the deep-note step |

For papers, the canonical `slug` is `{first-author-lastname}-{year}-{short-title-2-3-words}`, e.g. `hu-2021-lora`.

For blogs/threads, slug is `{author-handle}-{short-title-2-3-words}`, e.g. `karpathy-mup-scaling`.

---

## Part 6 — Cross-linking discipline

Every deep note must link:

- **Topic MOC(s)** via `topics: [topic-slug-1, topic-slug-2]` frontmatter AND a `[[topics/topic-slug]]` link in the body.
- **Author index** via `[[authors/firstname-lastname]]` in the body.
- **Adjacent notes** when relevant: if a paper cites or extends another paper already in the vault, link it via `related: [[papers/{slug}]]` in the body.

Topic MOCs (`30_Knowledge/topics/{topic}.md`) have this structure:

```markdown
---
type: moc
topic: flow-matching
last_updated: 2026-05-21
---

# Flow Matching

One-paragraph overview of the topic.

## Foundational
- [[papers/lipman-2023-flow-matching]] — TL;DR
- [[papers/liu-2023-rectified-flow]] — TL;DR

## Recent
- [[papers/...]] — TL;DR

## Related topics
- [[topics/diffusion-models]]
- [[topics/optimal-transport]]
```

Author indices (`30_Knowledge/authors/{slug}.md`) just list everything by
that author in the vault, with one-line context per item.

---

## Part 7 — Frontmatter contracts

### Paper note (`30_Knowledge/papers/{slug}.md`)

```yaml
---
type: paper
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: ["Edward Hu", "Yelong Shen", ...]
year: 2021
venue: ICLR 2022
url: https://arxiv.org/abs/2106.09685
rw_id: 01kb5cap1wy21zp37bc2rjj
topics: [efficient-finetuning, adaptation]
priority: high       # high | medium | low
read_state: queued   # queued | skimmed | deep | abandoned
relevance: ""        # free-form tag, e.g. "thesis-D1" if relevant to thesis vault
added: 2026-05-21
last_updated: 2026-05-21
---
```

### Blog note (`30_Knowledge/blogs/{slug}.md`)

```yaml
---
type: blog
title: "Why Flow Matching is Tractable"
author: "Lilian Weng"
url: https://lilianweng.github.io/...
rw_id: 01kb...
topics: [flow-matching, generative-models]
priority: medium
read_state: queued
added: 2026-05-21
last_updated: 2026-05-21
---
```

### Thread note (`30_Knowledge/threads/{slug}.md`)

```yaml
---
type: thread
title: "muP scaling — what most people get wrong"
author: "@karpathy"
url: https://twitter.com/...
rw_id: 01kb...
topics: [scaling, optimization]
priority: medium
read_state: queued
added: 2026-05-21
last_updated: 2026-05-21
---
```

### Topic MOC (`30_Knowledge/topics/{slug}.md`)

```yaml
---
type: moc
topic: flow-matching
last_updated: 2026-05-21
---
```

### Author index (`30_Knowledge/authors/{slug}.md`)

```yaml
---
type: author-index
name: "Lilian Weng"
last_updated: 2026-05-21
---
```

---

## Part 8 — File-naming conventions

- Daily raw inbox: `00_Inbox/raw/{YYYY-MM-DD}.md` (Worker appends)
- Category inbox: `00_Inbox/{papers|blogs|threads|newsletters}/{YYYY-MM-DD}.md` (you append during triage)
- Payload JSON: `00_Inbox/raw/payloads/{rw-id}.json` (Worker writes once, stays put)
- Processed raw bullets: `00_Inbox/raw/processed/{YYYY-MM-DD}.md`
- Deep note: `30_Knowledge/{papers|blogs|threads}/{slug}.md` (kebab-case slug per Part 5)
- Topic MOC: `30_Knowledge/topics/{topic-slug}.md`
- Author index: `30_Knowledge/authors/{firstname-lastname}.md`
- Blog draft workspace: `50_Writing/{blog-slug}/{index|research|draft}.md` (kebab-case blog-slug; see Part 14)

---

## Part 9 — Worker boundary

The Cloudflare Worker source at `90_Meta/worker/` is owned by the user.
You should:

- Edit it only when explicitly asked
- Not change deploy config (`wrangler.toml`) silently
- Flag if you notice the Worker's classification heuristics drifting from
  Part 5's rules — that's a sign one or the other needs updating

The Worker's responsibilities end at writing raw bullets + JSON sidecars
to the vault repo. **All categorization, deep-note creation, and
cross-linking happen in Claude's triage pass.** The Worker stays dumb on
purpose so heuristics can evolve without redeploying.

---

## Part 10 — Git policy

- Vault repo is private on GitHub
- Worker commits one-per-webhook to `main` directly via the GitHub Contents API
- `SessionStart` hook in `.claude/settings.json` runs `git pull` so the
  vault is fresh when you open Claude
- Triage passes batch into one commit at the end ("triage YYYY-MM-DD")
- Snapshot script in `90_Meta/scripts/snapshot.sh` for manual commits

---

## Part 11 — End-of-triage ritual

After processing the inbox:

1. Print a one-screen summary:
   - Items processed (count by category)
   - Deep notes created (with paths)
   - Topic MOCs created / updated
   - Author indices created / updated
   - Items rejected (with reason)
2. Run `git add -A && git commit -m "triage YYYY-MM-DD"` (commit, don't push — user pushes manually).
3. Update `30_Knowledge/_MOCs/dashboard-recent.md` with the newest items.

---

## Part 12 — Deep note body templates

Use these structures when creating notes under `30_Knowledge/{type}/{slug}.md`. The target lengths are **floors, not ceilings** — and they are deliberately generous. Default to the *fuller, more detailed* end of each range: a note that captures specifics (exact numbers, equations, hyperparameters, named methods, concrete examples) is far more valuable later than a terse one. When in doubt, write more. The only hard limit is the no-fabrication rule — depth must come from the fetched source, never from filler or invention. If a section is genuinely unsupported by the source, write `_not addressed by the source_` rather than padding.

**Where to spend the words (priority order):** the **Method and any derivations** are the point of the note — go deep there. Reconstruct the formulation, the core idea, the architecture/algorithm, and especially the **math: write out the key equations, define every symbol, and walk through the derivation steps** (objective → loss → update/sampling rule) rather than just stating the final loss. If the paper proves something, sketch the proof's logic. **Results can be summarized** — report the headline numbers and the most telling comparisons/ablations concisely; you do not need to transcribe every table. A good paper note is one you could re-implement the method from, even if you'd re-open the paper to check exact benchmark scores.

### Paper notes (target: 1200–2500+ words; go longer for rich papers)

```markdown
## TL;DR

(4–7 sentences) What the paper does, the core trick, why it matters, and
the headline result with its key number. Write this so a colleague who has
60 seconds gets the actual contribution — the mechanism and the evidence,
not just the topic. Name the method, state what it beats and by how much.

## Context & motivation

(1–2 paragraphs) The problem being attacked. What prior approaches failed
to solve or did inefficiently. The paper's stated contribution. Cite prior
work the paper itself cites, only where it matters for understanding the
contribution — don't fabricate citations.

## Method

### Problem formulation
Input / output / objective. What's being optimized.

### Core idea
1–2 sentences capturing the central insight.

### Architecture / algorithm
Each component and how they connect. **This is where the note earns its
keep — be thorough.** Use LaTeX-style math for every key equation (loss
functions, update rules, sampling procedures) and **define each symbol as
you introduce it**. Don't just quote the final loss — show how it's built:
the objective, the assumptions, the intermediate steps, and any derivation
or proof sketch the paper gives. Where the paper relies on a known result,
name it and state how it's used.
For example: $\mathcal{L} = \mathbb{E}_{x,t}[\| \epsilon - \epsilon_\theta(x_t, t) \|^2]$ — then explain what each term is and where it comes from.

### Derivations / why it works (when the paper has them)
Reconstruct the load-bearing math: the steps from setup to result, the key
inequality or identity, what assumption each step rests on. The goal is that
future-you understands *why* the method is correct, not just *what* it is.
Skip only if the paper is purely empirical (then write `_no derivation; empirical paper_`).

### Training procedure
Datasets used, optimizer, hyperparameters, schedule. Be specific —
"AdamW, lr=1e-4, cosine schedule, 200k steps" beats "standard setup".

### Inference / sampling (if generative)
How is generation done at test time — number of steps, guidance scales,
solver choice.

## Experimental setup

- Datasets: list explicitly
- Baselines: which methods, which papers
- Metrics: which numbers are being compared, what's "good"

## Key results

*Summarize — don't transcribe.* A few headline numbers with table/section
references are enough; the method sections above are where the depth goes.
- Headline metrics vs baselines (the one or two numbers that matter)
- Most surprising findings (positive or negative)
- Reported failure cases

## Ablations

What was ablated and what it revealed. This often matters more than the
headline number for understanding what's load-bearing.

## Limitations

Paper's own acknowledged limits + what an honest reader would flag.
Label which is which.

## Why it matters [analyst's view]

*This section is your synthesis, not paper claims.* What does this enable
downstream? Which directions does it open or close? What does it connect
to elsewhere in the vault?

## Open questions / things to verify

Anything that wasn't fully convincing. References you'd want to chase.
Experiments you'd want to see done.

## Connections

- Builds on: [[papers/...]]
- Extends to: [[papers/...]]
- Contrasts with: [[papers/...]]
- Topic MOCs: [[topics/...]]
- Author indices: [[authors/...]]

## Selected quotes (optional)

2–5 verbatim passages with section/page references, for direct citation later.

> "Quote text" — §3.2, p.5
```

### Blog notes (target: 500–1200 words; go longer for substantive technical posts)

```markdown
## TL;DR

(3–5 sentences) The post's central claim, the reasoning or evidence behind
it, and why it's worth remembering. Enough that you needn't re-read the post
to recall its actual argument, not just its subject.

## Context

(1–2 paragraphs) What prompted the post, intended audience, where it sits
in the author's broader work, and what debate or prior post it responds to.

## Core argument

The main claim laid out in full, with the key supporting evidence, examples,
and intermediate steps the author uses to get there. Capture the chain of
reasoning, not just the conclusion — but stay to the load-bearing parts.

## Notable details

- Specific technical points worth remembering (with the actual numbers /
  names / mechanisms, not just that they were mentioned)
- Examples / case studies the author uses, and what each illustrates
- Counter-intuitive claims with the author's reasoning for them
- Caveats or hedges the author flags

## Why it matters [analyst's view]

Connection to your interests / current work, and how it relates to other
notes in the vault. Where you agree, doubt, or want to dig further.

## Connections

- Topic MOCs: [[topics/...]]
- Related papers: [[papers/...]]
- Author index: [[authors/...]]

## Selected quotes (optional)

2–3 passages worth preserving verbatim.
```

### Thread notes (target: 100–300 words)

```markdown
## TL;DR

(1–2 sentences)

## Key claims

- 3–7 bullets capturing what the thread argues
- Quote the actual tweet language when the phrasing matters

## Context

(1 sentence) Author, when, what they were responding to.

## Why it matters [analyst's view]

(1–2 sentences)

## Connections

- Topic MOCs: [[topics/...]]
- Related papers / blogs: [[...]]
```

### Newsletter notes (target: 200–500 words)

```markdown
## TL;DR

(2–3 sentences)

## What it covered

Brief rundown of the main items / topics in this issue.

## Worth following up on

- Items, links, or papers mentioned that warrant a deeper read
- These can later become their own paper/blog notes after you save them
  to Readwise

## Connections

- Topic MOCs: [[topics/...]]
- Newsletter source: [[authors/...]]
```

---

## Part 13 — Gotchas

- **Don't auto-fetch the full content of every item.** Only papers and blogs warrant the Readwise API call. Threads have low information density per call; newsletters are usually summaries already.
- **Don't classify aggressively.** If unsure between blog and paper (e.g. a workshop note hosted on a personal site), default to blog and add a comment in the note. Reclassification is cheap; mis-filing into `papers/` pollutes the venue analysis.
- **Don't create a topic MOC for a topic with only one paper.** Wait for the second. Singleton MOCs are noise.
- **Don't deep-read during triage.** Triage is fast: fetch content, summarise, file, link. Deep reading is a separate session. The `read_state: queued | skimmed | deep` field tracks this.

---

## Part 14 — Writing workspace (`50_Writing/`)

The vault doesn't just capture reading — it feeds **writing**. `50_Writing/` is
where the user drafts blog posts, mining the knowledge base for material.

### Structure

```
50_Writing/
  README.md            # human-facing overview
  _template/           # canonical three-file scaffold — copy, never edit in place
    index.md
    research.md
    draft.md
  {blog-slug}/         # one subdirectory per blog the user wants to write
    index.md           # the brief: frontmatter + thesis, key messages, source notes
    research.md        # gathered material, every claim [[linked]] to a 30_Knowledge note
    draft.md           # outline → prose (the user's voice)
```

### The three files

- **`index.md`** (`type: blog-draft`) — control panel. Frontmatter: `title`,
  `slug`, `status`, `audience`, `angle` (one-sentence thesis), `target_length`,
  `topics`, `source_notes` (vault notes as `[[wikilinks]]`), `created`,
  `last_updated`. Body: thesis/hook, why-write-this, key messages, working titles,
  source notes, a running **status log**.
- **`research.md`** (`type: blog-research`) — the evidence base. Organised into
  core claims / supporting numbers / quotes / connections / gaps. **Every factual
  item links to its `30_Knowledge` source note.** Nothing here is written from
  memory — if it's not in the vault, mark `_needs note_` and queue the reading.
- **`draft.md`** (`type: blog-draft-body`) — outline first, then prose. Claude
  scaffolds the outline and gathers ammunition; the user writes (or asks Claude to
  draft named sections). Keep every factual claim traceable to `research.md`.

### Status lifecycle

`gathering` → `outlining` → `drafting` → `revising` → `published`
(tracked in `index.md` frontmatter; update it as the piece moves.)

### Rules

- **No fabrication, same as everywhere.** `research.md` is an evidence base that
  traces to vault notes; the gather step is `grep`-then-link, not recall.
- **Don't confuse with `30_Knowledge/blogs/`.** Those notes summarise things the
  user *read*; `50_Writing/` holds things the user intends to *write*.
- **The gather step deepens the vault.** If a draft exposes a gap (`_needs note_`),
  that's a signal to read & deep-note the missing source — writing and reading
  reinforce each other.
- **Don't ghost-write unprompted.** Default to scaffolding + research; only write
  prose into `draft.md` when the user asks.
- Published pieces may stay here (`status: published`) or move to `40_Archive/`.
