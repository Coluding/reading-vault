---
description: Write a one-page technical synthesis of a topic — main claims of selected vault papers, compared and combined
argument-hint: <topic>
---

# /synthesize `<topic>`

Produce a **one-page, technically dense synthesis** of a topic from the papers
already in the vault: surface each paper's main claims, **compare** the papers
(assumptions, methods, results, disagreements), and combine them into a single
coherent picture. The output is a durable note under `30_Knowledge/_synthesis/`.

This is **vault-mining, not reading**: the synthesis is built exclusively from
existing `30_Knowledge` deep notes. No web fetches, no Readwise API calls, no
recall from memory. If the vault's notes are too thin to support a claim, the
synthesis says so — that gap is the finding.

The topic is provided as the argument: `$ARGUMENTS`. If no topic is given, ask
for one.

## Procedure

1. **Resolve the topic.** Map `$ARGUMENTS` to a topic slug. Check
   `30_Knowledge/topics/` for an existing MOC (fuzzy-match: "flow matching" →
   `flow-matching.md`). The topic does **not** need an existing MOC — a
   frontmatter grep may still find papers. If neither a MOC nor any tagged
   paper exists, report that and stop; don't synthesize from nothing.

2. **Gather candidate papers.** Cast a wide net, then rank:
   ```bash
   # tagged directly
   grep -l "topic-slug" 30_Knowledge/papers/*.md
   # linked from the topic MOC
   grep -o "papers/[a-z0-9-]*" 30_Knowledge/topics/{topic}.md
   # adjacent topics listed under the MOC's "Related topics" — grep those too
   ```
   Also grep the topic term in paper bodies (`grep -li "flow matching"
   30_Knowledge/papers/*.md`) to catch papers that discuss the topic without
   being tagged. Blogs and threads may join as **supporting** sources when
   clearly relevant, but papers are the spine. For each candidate, read just
   the frontmatter + TL;DR to write a one-line description and a relevance
   rank.

3. **Let the user pick the papers.** Present the candidates with
   `AskUserQuestion`, **multiSelect: true**, default-recommending the most
   relevant ones:
   - One option per paper: label = short slug (e.g. `lipman-2023-flow-matching`),
     description = one-line TL;DR + why it's relevant.
   - Max 4 options per question and 4 questions per call — group candidates
     (e.g. "Core papers", "Adjacent/contrast papers") and use multiple
     questions, or a second `AskUserQuestion` call if there are more than 16.
   - If there are more candidates than you can present, present the top-ranked
     and say in the question text how many were cut and why.
   Everything the user leaves unticked is **excluded**. Proceed with the
   selection; if they select nothing, stop and ask what they'd rather cover.

4. **Read the selected notes in full** — the whole deep note, not just the
   TL;DR. Extract per paper: the main claims, the method's core idea, key
   numbers, stated limitations, and anything the note flags as unverified.
   Note where two papers address the same question (same benchmark, same
   theoretical object, competing formulations) — those are the comparison
   axes.

5. **Write the synthesis** to `30_Knowledge/_synthesis/{topic-slug}.md`
   (template below). Target **600–1000 words — one dense page**. The reader is
   very technical: use precise terminology, real equations from the notes when
   they carry the argument, and exact numbers with their source paper. Depth
   over breadth; no filler, no hedging boilerplate.

   If `_synthesis/{topic-slug}.md` already exists, **update it in place**
   (rewrite the body against the new selection, bump `last_updated`) rather
   than duplicating. A synthesis is a snapshot of the vault — regenerating it
   as the vault grows is the intended lifecycle.

6. **Cross-link.**
   - In the topic MOC (if it exists), add/refresh a line under a
     `## Syntheses` section: `- [[_synthesis/{topic-slug}]] — one-line scope
     (N papers, YYYY-MM-DD)`.
   - The synthesis itself links every paper it uses via `[[papers/...]]`.

7. **Commit.** `git add -A && git commit -m "synthesis: {topic-slug} ({N} papers)"`.
   Do not push. One commit per run.

8. **Print a short summary**: the note path, which papers went in, which were
   offered but unticked, and any gaps flagged (papers whose notes were too
   shallow to compare properly — candidates for a deep-read session).

## Note template

```markdown
---
type: synthesis
title: "Flow Matching — vault synthesis"
topic: flow-matching
papers: [lipman-2023-flow-matching, liu-2023-rectified-flow]
supporting: []            # blogs/threads used, if any
created: 2026-07-22
last_updated: 2026-07-22
---

# {Topic} — vault synthesis

## The picture in brief

(3–5 sentences) The state of the topic as told by these papers: the shared
problem, the main lines of attack, and where the frontier sits. Technical,
not promotional.

## Main claims by paper

One tight paragraph (or 2–4 bullets) per paper: the central claim, the
mechanism behind it, and the headline evidence with its number. Each entry
starts with the `[[papers/{slug}]]` link. Claims come from the vault note —
nothing from memory.

## Comparison

The load-bearing section. Compare along the axes the papers actually share:
- **Formulation / assumptions** — where they define the problem differently
  and what that buys or costs each of them.
- **Method** — the essential mechanical difference (equations when the notes
  have them).
- **Results** — head-to-head numbers where the notes report a common
  benchmark; note when they *aren't* comparable and why.
- **Disagreements** — claims that contradict each other, or where paper B's
  ablations undercut paper A's story. Name them explicitly.
A compact table is fine when ≥3 papers share axes; prose for the nuance.

## Synthesis [analyst's view]

*Interpretation, clearly separated from paper claims.* What the combined
evidence supports, what remains contested, which combination of ideas looks
promising, and what the vault should read next to close the gaps.

## Gaps

- Papers whose vault notes were too shallow to extract claims from
  (`_vault note too thin — deep-read needed_`), and known relevant work the
  vault lacks (`_needs note_`).

## Sources

- [[papers/...]] — role in this synthesis (core / contrast / supporting)
- [[topics/{topic-slug}]]
```

## Hard rules

- **Vault-only evidence.** Every claim traces to a `30_Knowledge` note (which
  traces to a real source). If a note doesn't support it, it doesn't go in —
  `_vault note too thin_` / `_needs note_` instead. Never pad from memory.
- **One page.** 600–1000 words. If the topic genuinely needs more, say so in
  the summary and suggest splitting into sub-topic syntheses — don't sprawl.
- **Comparison is mandatory.** A synthesis that just stacks per-paper TL;DRs
  has failed; the value is in the axes where the papers meet.
- **`[analyst's view]` labelling** for everything that is your interpretation
  rather than a paper's claim — same discipline as deep notes.
- **`_synthesis/` is not a Knowledge category.** Like `_MOCs/`, it's a derived
  layer over the five note types — the Worker never writes here, and synthesis
  notes never replace deep notes.
- **Update-in-place on re-run** for the same topic; the git history preserves
  old snapshots. Never delete.
