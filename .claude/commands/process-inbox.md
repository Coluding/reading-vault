---
description: Triage the raw inbox into category inboxes and deep notes
---

# /process-inbox

Read every unprocessed entry in `00_Inbox/raw/*.md` (any file not present
in `00_Inbox/raw/processed/`), classify each entry, route to the
appropriate category inbox, then create deep notes in `30_Knowledge/`.

## Procedure

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

3. **For each non-reject item, create a deep note** at `30_Knowledge/{type}/{slug}.md`:
   - Slug per `CLAUDE.md` Part 5
   - Frontmatter per `CLAUDE.md` Part 7
   - For papers and blogs: fetch full content via the Readwise Document Export API:
     ```
     GET https://readwise.io/api/v3/list/?id={rw-id}
     Authorization: Token {READWISE_TOKEN}
     ```
     (The user has the token set as `READWISE_TOKEN` env var or in `.env`. Read it from there; do not hardcode.)
   - For threads and newsletters: use the `summary` field from the webhook payload — do not call the API.
   - Body sections:
     - One-line TL;DR (your synthesis from fetched content)
     - 3–5 key claims (quoted or paraphrased from the content, with attribution)
     - "Why it matters" — your synthesis, labelled as such
     - "Open questions" — what you'd want to verify
     - "Related" — `[[wikilinks]]` to existing vault notes you can find via grep on topics/authors

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
