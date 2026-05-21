---
type: moc
last_updated: 2026-05-21
status: living
---

# Dashboard — Deep-Read Queue

Items flagged `priority: high` and `read_state: queued` — what you should
read next.

## Papers

_Nothing yet._

## Blogs

_Nothing yet._

## Threads

_Nothing yet._

---

## How this dashboard works

Rewritten at the end of `/process-inbox`. Items leave this list when their
`read_state` is updated to `skimmed`, `deep`, or `abandoned`.

To update a note's read state, edit the frontmatter directly:

```yaml
read_state: deep   # was: queued
```

Items here are ordered by `added` date ascending (oldest queued first) so
the backlog doesn't get lost behind newer arrivals.
