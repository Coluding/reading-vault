---
type: moc
last_updated: 2026-05-21
status: living
---

# Dashboard — Orphans

Notes with no inbound `[[wikilinks]]`. Usually means one of:

- The note's topic MOC was never created (singleton-topic)
- The author index is missing
- The note was mis-filed and doesn't belong in its current category

## Orphan papers

_Nothing yet._

## Orphan blogs

_Nothing yet._

## Orphan threads

_Nothing yet._

---

## How to fix an orphan

1. Check the note's `topics:` frontmatter — does each topic have a MOC?
2. Check the note's authors — do they have author indices?
3. If both exist but no link, the MOCs are stale. Re-run `/process-inbox` or manually link.
4. If the note doesn't fit any existing topic, consider whether it belongs in the vault at all. Move to `40_Archive/` if not.
