---
description: Build a single-file HTML dashboard, slide deck, or React SPA visualizing a topic or the whole vault
---

# /visualize

Generate a self-contained visualization artifact for either a specific topic or the entire vault. Output is a single HTML file saved under `60_Visualizations/` — no build step, no external data fetches at runtime, works offline.

## Usage

- `/visualize` — global view across the whole vault
- `/visualize {topic-slug}` — focused on one topic (e.g. `/visualize flow-matching`)
- Pass `--as=dashboard` (default), `--as=presentation`, or `--as=react`
- Pass `--filter={priority|recent|deep|orphans}` to scope content further (optional)

## Procedure

### 1. Parse arguments

Extract:
- `topic` — first non-flag positional arg, or `"all"` if none
- `format` — `dashboard | presentation | react`, default `dashboard`
- `filter` — optional content filter

If a `topic` is given but `30_Knowledge/topics/{topic}.md` doesn't exist, list available topic MOCs (`ls 30_Knowledge/topics/`) and stop. Don't invent a topic.

### 2. Gather content

**Topic mode** (specific topic):
- Read `30_Knowledge/topics/{topic-slug}.md` (the MOC).
- For each `[[papers/...]]`, `[[blogs/...]]`, `[[threads/...]]` link in the MOC, read the full deep note.
- For each author referenced in those notes, read `30_Knowledge/authors/{author-slug}.md`.
- Find related topics from `[[topics/...]]` links in the MOC body — collect their names but don't recurse into their notes.

**Global mode** (`topic == "all"`):
- List all topic MOCs: `find 30_Knowledge/topics -maxdepth 1 -name '*.md'`
- List all deep notes: `find 30_Knowledge/{papers,blogs,threads} -maxdepth 1 -name '*.md'`
- List all author indices.
- Read **frontmatter + first 30 lines** of each deep note (full body is too much for a global view). Extract: `title`, `authors`, `year`, `topics`, `priority`, `read_state`, `relevance`, `added`, plus the TL;DR if present.

Apply `--filter` if set:
- `priority` → only `priority: high`
- `recent` → only items added in the last 14 days
- `deep` → only `read_state: deep`
- `orphans` → notes with no inbound `[[wikilinks]]` (check by grepping the vault)

### 3. Build the data model

Construct an in-memory JSON object before writing any HTML:

```js
{
  "generated_at": "2026-05-22T14:30:00Z",
  "scope": { "type": "topic" | "global", "name": "flow-matching" | "all" },
  "stats": {
    "papers": 12,
    "blogs": 4,
    "threads": 7,
    "topics": 8,
    "authors": 23,
    "queued": 5,
    "deep_read": 9
  },
  "papers": [
    {
      "slug": "hu-2021-lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "authors": ["Edward Hu", ...],
      "year": 2021,
      "venue": "ICLR 2022",
      "url": "https://arxiv.org/abs/2106.09685",
      "rw_id": "01kb...",
      "topics": ["efficient-finetuning", "adaptation"],
      "priority": "high",
      "read_state": "deep",
      "tldr": "...",       // verbatim from the note body
      "key_claims": [...], // pulled from the "Key results" section if present
      "added": "2026-05-21"
    },
    ...
  ],
  "blogs": [...],
  "threads": [...],
  "topics": [
    {
      "slug": "flow-matching",
      "title": "Flow Matching",
      "paper_slugs": [...],   // outbound links
      "related_topics": [...]
    }
  ],
  "authors": [
    { "slug": "edward-hu", "name": "Edward Hu", "paper_slugs": [...] }
  ]
}
```

This object is embedded inline in the HTML as a `<script type="application/json" id="vault-data">…</script>` block. **All data goes in once at generation time.** No runtime fetches.

### 4. Generate the artifact

#### Format = `dashboard` (default)

Single HTML file with:
- **Tailwind via CDN** (`https://cdn.tailwindcss.com`) for styling
- **Chart.js via CDN** (`https://cdn.jsdelivr.net/npm/chart.js`) for graphs
- **Vanilla JS** for interactivity — no framework

Layout:
1. **Hero strip** — vault scope (topic name or "Reading Vault"), generation date, summary stats as large numbers
2. **Topic grid** — clickable cards, each with topic name, paper count, list of top-3 paper titles
3. **Reading queue panel** — items with `read_state: queued` sorted by `priority desc, added asc`, showing title + author + reason-to-read snippet
4. **Author leaderboard** — top 10 authors by paper count, with paper titles on hover
5. **Timeline chart** — papers added per week (Chart.js bar chart)
6. **Recent additions feed** — last 10 items added, with TL;DRs

Search bar at the top should filter the topic grid and reading queue live (vanilla JS, no framework needed).

#### Format = `presentation`

Single HTML file using **Reveal.js via CDN** (`https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/`). Slides authored in `<section data-markdown>` blocks for readability.

Default slide outline:
1. **Title slide** — vault scope + generation date + headline stats
2. **Overview** — bar chart of papers per topic (use Chart.js inside a slide; init in `Reveal.on('ready')`)
3. **Per-topic slides** — one slide per topic, each listing 3–5 papers with their TL;DRs (bulleted, citing rw-id)
4. **Author highlights** — top contributors and their papers
5. **Reading queue** — what's next to read deeply
6. **Connections** — text representation of how topics link to each other (use a Mermaid graph if it makes sense; Mermaid via CDN works in Reveal slides)
7. **Open questions** — aggregated from `## Open questions` sections across deep notes

For topic mode, replace step 3 with deep-dive slides for that single topic (one slide per major paper).

#### Format = `react`

Single HTML file with React 18 + Babel standalone via CDN — no build step, but full JSX. Pattern:

```html
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://cdn.tailwindcss.com"></script>
<script type="application/json" id="vault-data">…</script>
<div id="root"></div>
<script type="text/babel">
  const data = JSON.parse(document.getElementById('vault-data').textContent);
  function App() { … }
  ReactDOM.createRoot(document.getElementById('root')).render(<App />);
</script>
```

Components to include (keep them all in one file):
- `<FilterBar />` — search + topic/priority/read_state filters
- `<StatsHeader />` — summary numbers
- `<TopicGrid />` — clickable cards; selecting one filters the paper list
- `<PaperList />` — virtualized only if N > 100 (otherwise just `.map`); each `<PaperCard />` expands to show TL;DR and key claims
- `<AuthorPanel />` — author leaderboard with click-to-filter
- `<Timeline />` — additions over time (use a tiny SVG chart written inline, no Chart.js needed)

State via `useState` + `useMemo`. No Redux, no router, no router required.

### 5. Content rules (CRITICAL)

- **No fabrication.** Every paper TL;DR, claim, or quote in the visualization must come verbatim or near-verbatim from the deep note. If a note has `read_state: queued` and a stub body, show "_(not yet processed — open the source to read)_" instead of inventing.
- **Cite `rw_id` on every paper card** so the source is traceable back to Readwise.
- **Preserve `[analyst's view]` labels.** If a deep note labels a section as analyst synthesis (not paper claim), the visualization must keep that distinction visible — e.g. a small tag/badge near the rendered text.
- **No external data at runtime.** All paper data embedded inline. No `fetch()` calls in the JS. CDN script tags are fine; data endpoints are not.
- **Use the frontmatter fields semantically:**
  - `priority` → visual emphasis (border color, badge)
  - `read_state` → grouping or filter
  - `relevance` → optional label (e.g. "thesis-D1")
  - `topics` → grouping
  - `added` → timeline placement

### 6. Save the output

Path: `60_Visualizations/{scope}-{format}-{YYYY-MM-DD}.html`

Where `scope` is the topic slug or `all`. Examples:
- `60_Visualizations/all-dashboard-2026-05-22.html`
- `60_Visualizations/flow-matching-presentation-2026-05-22.html`
- `60_Visualizations/transformers-react-2026-05-22.html`

If the same `scope-format-date` triple already exists, append `-v2`, `-v3`, etc.

### 7. Report

After writing, print:
- The absolute path of the generated file
- A one-line command to open it: `xdg-open <path>` (Linux) or `open <path>` (macOS)
- Summary stats: count of papers/blogs/threads/topics rendered
- Anything notable: missing data, stubbed sections, topics that had no content

**Do not commit the visualization automatically.** Let the user inspect it first; they'll commit if they want to keep it.

## Hard rules (recap)

- Single HTML file output, all data inline, works offline.
- No fabrication — visualizations only show what's in the vault's deep notes.
- Cite rw-id on every paper. Preserve `[analyst's view]` labelling.
- If `--filter` produces an empty set, say so and stop (don't generate an empty shell).
- If the requested topic doesn't exist, list available topics and stop.
