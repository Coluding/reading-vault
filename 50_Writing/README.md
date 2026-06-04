# 50_Writing — Blog drafting workspace

Where reading turns into writing. Each blog you want to write gets its **own
subdirectory** under `50_Writing/{blog-slug}/`, scaffolded with three files:

| File | Purpose |
|---|---|
| `index.md` | The brief — frontmatter (title, status, audience, angle, source notes) + working-title ideas, the thesis/hook, and key messages. The control panel for the piece. |
| `research.md` | The gathered material — facts, claims, numbers, and quotes pulled from `30_Knowledge/**`, **every item linked back to its source note** via `[[wikilinks]]`. No fabrication: everything here traces to a vault note (which traces to the original source). |
| `draft.md` | The actual blog post — outline first, then prose. This is *your* writing; Claude scaffolds the outline and gathers ammunition, you write the post (or ask Claude to draft sections). |

## Status lifecycle (in `index.md` frontmatter)

`gathering` → `outlining` → `drafting` → `revising` → `published`

## How to use it

- **Start a blog:** "start a blog on {topic}" — Claude creates the subdir from `_template/`, sets the angle, and runs the first research pass.
- **Gather more:** "gather research for the {slug} blog" — Claude re-scans `30_Knowledge` for relevant notes and appends to `research.md`.
- **Draft:** "outline / draft the {slug} blog" — Claude works `draft.md` from the research.

The vault's reading pipeline (inbox → triage → deep notes → MOCs) feeds this
directory: the deeper the knowledge notes, the richer `research.md` can be without
re-reading sources.

## Conventions

- Blog slug = kebab-case topic or working title, e.g. `predictive-processing`, `why-prediction-is-all-you-need`.
- `_template/` holds the canonical three-file scaffold — copy, don't edit it in place.
- Drafts here are **not** the same as `30_Knowledge/blogs/` notes: those summarize things you *read*; these are things you intend to *write*.
- Published posts can stay here (status `published`) or be archived to `40_Archive/` once live elsewhere.
