---
description: Ingest a company/research-org website into 30_Knowledge/websites and cross-link it to papers, blogs, and topics
argument-hint: <website-url>
---

# /ingest-website `<url>`

Parse a **company / research-org website** (usually a lab, startup, or product
site that publishes research) and turn it into a durable `website` note under
`30_Knowledge/websites/`, cross-linked into the vault's existing papers, blogs,
topics, and author indices.

Unlike papers/blogs/threads/newsletters, **websites do not arrive via the
Readwise Worker inbox** — there is no rw-id and no payload sidecar. This skill is
the ingestion path: it is triggered manually with a URL and fetches content
**live via the `WebFetch` tool** (and `WebSearch` when a page won't load or you
need to confirm who the org is). Everything else — the no-fabrication rule,
cross-linking discipline, topic-MOC and author-index upkeep — is the same as
`/process-inbox`.

The URL is provided as the argument: `$ARGUMENTS`. If no URL is given, ask for one.

## Procedure

0. **Resolve the target.** Take the URL from `$ARGUMENTS`. Normalise it (add
   `https://` if missing). This is the org's site.

1. **Fetch the site.** Use `WebFetch` on the given URL first. Then fetch the
   pages that actually describe the research — try, in order of usefulness, the
   ones that exist:
   - `/research`, `/publications`, `/papers`, `/work`
   - `/blog`, `/news`, `/updates`
   - `/about`, `/team`, `/people`
   - `/` (landing page) for the mission / one-liner

   Fetch **2–5 pages** — enough to characterise what they do and what they've
   published. If a page 404s or is JS-only and returns nothing useful, skip it;
   use `WebSearch` (`"<org name>" research`) to fill gaps about who they are.
   **Record every URL you actually fetched** — it goes in the note's `## Source`
   section. If you cannot fetch the site at all, report that and stop; do not
   invent an org profile.

2. **Extract the org profile.** From the fetched content pull:
   - **Org name** and one-line description (their own words where possible).
   - **What they do**: product vs. pure research, stage/funding if stated,
     domain.
   - **Research agenda**: the themes/directions they work on.
   - **Notable work**: named papers, projects, model/product releases they
     highlight — with links. Prefer their own list over your recollection.
   - **People**: founders / key researchers named on the site.

   **No fabrication.** Every stated fact must come from the fetched pages (or a
   `WebSearch` result you cite). Your interpretation goes only in the
   `[analyst's view]` section. If the site doesn't address a template section,
   write `_not addressed by the source_`.

3. **Choose the slug.** `{org-slug}` in kebab-case, e.g. `physical-intelligence`,
   `deepmind`, `sakana-ai`. If the vault already has a `websites/{org-slug}.md`
   for this org, **update it in place** (bump `last_updated`, add new findings)
   rather than creating a duplicate. Only append a `-{descriptor}` when one org
   genuinely needs two distinct site notes.

4. **Cross-link into the existing vault — this is the point of the note.** Before
   writing, search the vault for what this org already connects to:
   ```bash
   # papers/blogs by their people or about their work
   grep -rli "physical intelligence" 30_Knowledge/papers 30_Knowledge/blogs
   # each named person → existing author index or authorship
   grep -rli "sergey levine" 30_Knowledge/authors 30_Knowledge/papers 30_Knowledge/blogs
   # topics they work on → existing MOCs
   ls 30_Knowledge/topics/ | grep -iE 'robot|vla|imitation'
   ```
   Use the hits to populate the note's `## Notable work`, `## People`, and
   `## Connections` sections with real `[[wikilinks]]` (`[[papers/...]]`,
   `[[blogs/...]]`, `[[authors/...]]`, `[[topics/...]]`). Only link notes that
   actually exist — a wikilink to a missing note is fine only if you flag it as a
   reading suggestion, not as an existing connection.

5. **Write the deep note** at `30_Knowledge/websites/{slug}.md`:
   - Frontmatter per `CLAUDE.md` Part 7 (website contract).
   - Body per `CLAUDE.md` Part 12 (website template). Target **400–900 words** —
     enough to capture their agenda and every real vault connection; go longer
     for orgs with a large, relevant body of work.

6. **Update topic MOCs (bidirectional link).** For each topic in the note's
   `topics:` field that has an existing MOC, add the org under an
   `## Organizations` section (create that section if absent):
   ```markdown
   ## Organizations
   - [[websites/physical-intelligence]] — VLA / robot foundation models; ships π-series policies.
   ```
   Follow the singleton rule: don't create a brand-new topic MOC just for one
   website. If the org's only topic has no MOC yet, leave it — the link from the
   website note outward is enough until a second item joins the topic.

7. **Update author indices.** For each person you linked who already has an index
   in `30_Knowledge/authors/`, add a one-line entry noting their affiliation with
   this org and the `[[websites/{slug}]]` backlink. Do **not** create new author
   indices from a website alone — wait until the vault holds an actual paper/blog
   by them.

8. **Update `30_Knowledge/_MOCs/dashboard-recent.md`** — add the website under a
   `### Websites` heading in the current week's block, with a one-line TL;DR.

9. **Commit.** `git add -A && git commit -m "ingest website: {org-name}"`. Do not
   push — the user pushes manually. (One commit per skill run, like triage.)

10. **Print a short summary**: the note path, which papers/blogs/topics/authors
    it linked to, and any `_needs note_` reading suggestions the site surfaced
    (research the org highlighted that the vault doesn't cover yet).

## Hard rules

- **No fabrication.** If you didn't fetch it (or cite a `WebSearch` result), you
  don't know it. Don't guess funding, headcount, or publications from memory.
- **Websites bypass the inbox.** No rw-id, no payload JSON, no `00_Inbox` routing
  — this skill writes straight to `30_Knowledge/websites/`.
- **Cross-link only to notes that exist.** Flag missing-but-relevant sources as
  `_needs note_` reading suggestions instead of dead links presented as fact.
- **Don't create singleton topic MOCs or author indices** off a website alone —
  same discipline as `/process-inbox`.
- **Websites go stale.** Set `read_state` honestly (`skimmed` after a triage-style
  pass) and always stamp `last_updated` so a re-fetch later is easy to spot.
