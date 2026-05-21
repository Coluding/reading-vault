---
type: moc
last_updated: 2026-05-21
status: living
---

# Dashboard — Recent

Most recent items added to the vault, regardless of category. Refreshed by
`/process-inbox`.

## This week

_Nothing yet — Worker has not received any webhooks._

## Last week

_Nothing yet._

## Older

_Nothing yet._

---

## How this dashboard works

This file is rewritten at the end of every `/process-inbox` run. It shows:

- **This week**: items added in the last 7 days, grouped by category.
- **Last week**: items added 7–14 days ago.
- **Older**: rolled-up by month.

For deeper queries see the other dashboards in `30_Knowledge/_MOCs/`:

- `dashboard-deep-read.md` — items marked `priority: high` and `read_state: queued`
- `dashboard-orphans.md` — notes with no inbound links (potentially mis-filed)
