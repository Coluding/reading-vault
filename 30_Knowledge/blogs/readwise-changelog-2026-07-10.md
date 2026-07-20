---
type: blog
title: "Readwise & Reader Changelog — July 10, 2026"
author: "Readwise & Reader Changelog"
url: https://docs.readwise.io/changelog#july-10-2026
rw_id: 01kxeh89ntn6677q8hz9vkkj48
topics: [reading-tools]
priority: low
read_state: queued
added: 2026-07-14
last_updated: 2026-07-14
---

## TL;DR

A features-and-fixes release for Readwise Reader: Markdown file uploads, a Daily Review completion endpoint in the public API, a `language` parameter for saves, plus text-to-speech, MCP, chat-focus, and CLI bug fixes.

## What it covered

- **Markdown uploads** — Markdown files can now be uploaded to Reader like EPUBs/PDFs (drag-and-drop, file picker, or mobile share). Title comes from the file; the first image becomes the cover.
- **Daily Review API** — the public API can now mark a Daily Review complete (incl. past days), useful for personal dashboards/homepages built on Readwise.
- **Article Language Control** — a `language` parameter added to the Reader API and MCP server; a specified language now sticks and isn't overridden by automatic detection.
- **Text-to-speech fix** — resolved a bug where pressing play could fail with a server error when audio for other listeners was being generated concurrently.
- **Improved MCP responses** — the MCP server's document-list tool no longer returns full document bodies by default (they could be 100,000+ chars and clog an AI's context window); browsing now returns lightweight responses and burns far fewer tokens.
- **Chat focus fix** — pressing `esc` in the Ghostreader web chat panel returns focus to the document.
- **CLI hangs fix** — Readwise CLI v0.5.9 released, fixing hangs.

## Why it matters [analyst's view]

Two items are directly relevant to this vault's own tooling: **Markdown uploads** make it trivial to push vault notes into Reader, and the **MCP token-bloat fix** (document-list no longer dumps 100k-char bodies) matters for any agent — including this one — that browses the Reader library programmatically. The `language` parameter is a small quality-of-life win for multilingual saves.

## Connections

- Author index: [[authors/readwise-reader-changelog]]
