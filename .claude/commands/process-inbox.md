---
description: Triage the raw inbox into category inboxes and deep notes
---

# /process-inbox

Read every unprocessed entry in `00_Inbox/raw/*.md` (any file not present
in `00_Inbox/raw/processed/`), classify each entry, route to the
appropriate category inbox, then create deep notes in `30_Knowledge/`.

## Procedure

0. **Pull latest state.** Run `git pull --quiet --rebase --autostash` to fetch any Worker-written commits that landed since session start. The `SessionStart` hook already pulled once on open, but if the user has saved more papers to Readwise during this session, those commits are only on GitHub until this step runs. If `git pull` fails (offline, conflict), report it and stop — don't triage stale state silently.

1. **Find unprocessed raw files.** List `00_Inbox/raw/*.md` (excluding the `processed/` subdir). Filter out any date that already exists in `00_Inbox/raw/processed/`.

2. **For each unprocessed bullet:**
   - Parse the bullet to extract the `rw-id`.
   - Read `00_Inbox/raw/payloads/{rw-id}.json` for full structured fields.
   - **Classify** using the rules in `CLAUDE.md` Part 5:
     - `paper` — arxiv/openreview/proceedings/PDFs
     - `thread` — twitter/x.com
     - `newsletter` — substack/buttondown emails
     - `blog` — known AI/math/CS blogs or default article
     - `reject` — clearly off-topic (promo emails, unrelated news)
   - **Route**: append the bullet (verbatim) to `00_Inbox/{type}/{YYYY-MM-DD}.md`. For rejects, just note `[reject: {reason}]` in the processed file and move on — no deep note.

3. **For each non-reject item, create a substantial deep note** at `30_Knowledge/{type}/{slug}.md`:
   - Slug per `CLAUDE.md` Part 5
   - Frontmatter per `CLAUDE.md` Part 7
   - **Body per `CLAUDE.md` Part 12** — these notes are the durable record; depth matters. Target lengths are **floors** — default to the fuller, more detailed end and go longer when the source warrants it. **Spend the words on Method and derivations** (write out and explain the key equations, define symbols, walk the derivation/proof logic); **results can be summarized** (a few headline numbers, not every table). **Method sections must pass the self-sufficiency test** (Part 12): every equation with all symbols defined in place and the *why* of each term, losses decomposed term-by-term, choices motivated — readable without opening the paper. If the fetched source lacks the method details (supplement-only, abstract-only), flag it prominently at the top and list what's missing:
     - **Paper**: 1200–2500+ words across the full template (TL;DR, context, method with subsections + derivations, experimental setup, results summarized with key numbers, ablations, limitations, analyst's synthesis, open questions, connections, optional quotes). Use LaTeX math for every key equation and define its symbols. Aim for a note you could re-implement the method from.
     - **Blog**: 500–1200 words (TL;DR, context, core argument, notable details, why it matters, connections, optional quotes). Capture the chain of reasoning and concrete details, not just the conclusion.
     - **Thread**: 100–300 words (TL;DR, key claims, context, why it matters, connections).
     - **Newsletter**: 200–500 words (TL;DR, what it covered, follow-ups, connections).
   - For papers and blogs: fetch full content via the Readwise Document Export API:
     ```
     GET https://readwise.io/api/v3/list/?id={rw-id}
     Authorization: Token {READWISE_TOKEN}
     ```
     (The user has the token set as `READWISE_TOKEN` env var or in `.env`. Read it from there; do not hardcode.) The full text is what makes a deep note possible — without it, you'd be writing a stub.
   - For threads and newsletters: use the `summary` field from the webhook payload — do not call the API.
   - **No fabrication**: every fact, quote, or number must trace to the fetched content. Your synthesis goes in sections labelled `[analyst's view]`. If the source doesn't address a template section (e.g. a paper with no ablations), write `_not addressed by the source_` rather than inventing.

4. **Update topic MOCs**: for each topic in the new note's `topics:` field, append a `[[link]]` to `30_Knowledge/topics/{topic}.md` under the appropriate section. Create the MOC if missing (skeleton per `CLAUDE.md` Part 6). **But:** skip creating a MOC for a topic that would only have one paper — wait for the second.

5. **Update author indices**: for each author, append a one-line entry to `30_Knowledge/authors/{author-slug}.md`. Create the index if missing.

6. **Mark processed**: move all processed bullets from `00_Inbox/raw/{date}.md` to `00_Inbox/raw/processed/{date}.md`. The payload JSONs stay in `00_Inbox/raw/payloads/` permanently — they're the durable record.

7. **Update `30_Knowledge/_MOCs/dashboard-recent.md`** with the newest items.

8. **End-of-triage summary** — print one screen:
   - N items processed (X papers, Y blogs, Z threads, …)
   - Deep notes created (paths)
   - Topic MOCs created/updated
   - Author indices created/updated
   - Items rejected (with reason)

9. **Single commit at the end**: `git add -A && git commit -m "triage YYYY-MM-DD"`. Do not push — user pushes manually.

## Hard rules

- **Never fabricate paper claims, numbers, or quotes.** If the API call fails, write `_needs verification_` in the relevant section and stop on that item.
- **Cite the rw-id** in every deep note's frontmatter so the source is recoverable.
- **Don't deep-read during triage.** Triage = fetch, summarise, file, link. Deep reading is a separate session.
- **Don't promote singleton-topic MOCs.** Wait for the second paper on a topic.
