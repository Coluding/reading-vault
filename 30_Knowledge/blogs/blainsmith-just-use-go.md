---
type: blog
title: "Just Fucking Use Go"
author: "Blain Smith"
url: https://blainsmith.com/articles/just-fucking-use-go/
rw_id: 01ksnds2g2w3fd3z4x82nfg4hc
topics: [software-engineering, go-lang, backend-architecture]
priority: low
read_state: skimmed
relevance: ""
added: 2026-05-27
last_updated: 2026-05-27
---

# Just Fucking Use Go

**TL;DR** — A profanity-laden opinion rant arguing that Go is the correct default for backend web apps and that most teams overcomplicate things with framework stacks, microservices, Kubernetes, and rewrites. The thesis: Go's deliberate boringness (small language, one formatter, no clever abstractions), a "stdlib *is* the framework" philosophy, cheap goroutines, dead-simple dependency management (`go.mod`/`go.sum`, no `node_modules`), and single-static-binary deployment together make it the highest-leverage boring choice. Not an AI/math piece — pure software-engineering polemic — but on-topic as a CS/backend-systems read.

## Context

A short (1,583-word) blog post on `blainsmith.com`, published 2026-05-08. It's an entry in the long-running "use boring technology" genre (explicitly riffing on the "just use HTML" frontend version). Audience: backend/full-stack engineers reaching for heavyweight stacks (Rails, Django, Express, Next.js) or premature microservices/Rust rewrites. No data, benchmarks, or citations — it's an argument from developer ergonomics, not measurement.

## Core argument

**Boring is the point.** Go has "structs, functions, interfaces, goroutines, and channels. That's it." No decorators, metaclasses, macros, traits, or monads. The payoff is legibility: a junior can read the principal's two-year-old code, `gofmt` ends formatting debates, and nobody can smuggle "a seventeen-layer abstraction" past the language.

**The standard library is the framework.** `net/http` is both server and client; `database/sql`, `encoding/json`, `html/template`, `context`, `io.Reader`/`io.Writer`, `pprof`, and `go test` (with `-race`, `-bench`, `-cover`) all ship in-box. The author shows a full embedded-template web app and a Postgres-backed CRUD handler — request `context` wired through to the SQL query so a closed connection cancels the query — each fitting "on one screen."

**Concurrency without ceremony.** Goroutines (~2KB stacks, multiplexed onto OS threads) let you "spawn a hundred thousand on a laptop"; channels are typed pipes; `sync.Mutex` + the race detector cover shared state. A parallel HTTP fetcher is a few lines with no async/await.

**Operational simplicity is the real sell.** Dependencies are two files (`go.mod` + cryptographic `go.sum`), no lockfile drift, `go mod vendor` for offline builds. Deployment is "a copy command": cross-compile a ~12MB static binary, `scp` it, restart a systemd unit — "No Dockerfile ... No Kubernetes manifest. No Helm chart." Write the monolith; split into separate repos later only if you ever need to (the interfaces already exist).

## Notable details

- Counter-arguments dispatched: `if err != nil` is framed as a feature (forces handling every failure point vs. hiding it in try/catch until 2am); generics "landed in 1.18 ... they're fine."
- The whole piece is rhetorical/aggressive ("Hey, dipshit"; "you absolute walnut") — entertainment value over rigor; zero acknowledgement of cases where Go is a poor fit (heavy generic-data-structure work, certain ML/numeric workloads, ecosystems where a mature framework genuinely saves time).

## Why it matters [analyst's view]

Low durable value for this vault — it's culturally adjacent to the systems/GPU material the owner is collecting (see [[authors/lukas-bierling]]'s CUDA notes) but adds no technical substance beyond well-worn Go folklore. Worth keeping only as a tidy reference for the "boring monolith + single binary" argument if it ever comes up in a tooling decision. No second post on Go programming exists in the vault yet, so the `go-lang` / `software-engineering` topics stay singletons — no MOC created (per the don't-promote-singletons rule). If more systems/backend-engineering reading lands, revisit and spin up a `software-engineering` MOC then.

## Connections

- Topic MOCs: _none created — singleton topics (`go-lang`, `software-engineering`, `backend-architecture`); awaiting a second item_
- Author index: [[authors/blain-smith]]

## Selected quotes

> "You know why Go feels boring? Because it is, and that's the goddamn point."

> "Stop looking for a framework, you absolute walnut. The standard library is the framework."

> "A 12MB statically linked binary and a 20-line systemd unit file is a production deployment. It will outlive your career."
