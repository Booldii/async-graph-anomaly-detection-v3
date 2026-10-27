# Docker

Two separate compose files - different lifecycles:

- `docker-compose.infra.yml` - stateful (Redpanda + Redis, with volumes). Started once per work
  session and left running; rebuilding the app stack should never touch these volumes.
- `docker-compose.app.yml` - stateless app code (this project's own services). Rebuilt/restarted
  on every code change, independently of the infra stack above.

Both share one external Docker network, `onchain-net` (declared in `docker-compose.infra.yml`,
referenced as `external: true` in `docker-compose.app.yml`).

## Running the full demo stack

One-shot (infra up, artifacts fetched if missing, app stack built and started):

```bash
make demo
```

Or the individual steps, same order (see `Makefile` at the repo root for all targets -
`infra-up`, `fetch-artifacts`, `demo-up`, `demo-down`/`down`, `infra-down`):

```bash
make infra-up          # Redpanda + Redis (once per work session)
make fetch-artifacts   # idempotent - skips files already present locally
make demo-up           # replay_producer, inference_consumer, alert_panel, alert_dashboard
```

Then open `http://localhost:8501` for the alert dashboard (`http://localhost:8000/alerts` for
the raw backend JSON). `make down` stops the app stack only - infra is left running on purpose
(see "Two separate compose files" above); `make infra-down` stops that too.