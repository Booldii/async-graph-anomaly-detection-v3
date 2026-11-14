# Docker

Two separate compose files, on purpose - different lifecycles:

- `docker-compose.infra.yml` - stateful (Redpanda + Redis, with volumes). Started once per work
  session and left running; rebuilding the app stack should never touch these volumes.
- `docker-compose.app.yml` - stateless app code (this project's own services). Rebuilt/restarted
  on every code change, independently of the infra stack above.

Both share one external Docker network, `onchain-net` (declared in `docker-compose.infra.yml`,
referenced as `external: true` in `docker-compose.app.yml`).

## Running the app stack

One-shot (infra up, artifacts fetched if missing, app stack built and started) - pick one of two
mutually exclusive event sources:

```bash
make replay   # replays a recorded historical sample - no API key needed
```

```bash
make live     # subscribes to the real Ethereum mainnet feed via Alchemy (needs
              # ALCHEMY_API_KEY in the project's .env - a free Alchemy account is enough)
```

Or the individual steps, same order (see `Makefile` at the repo root for all targets -
`infra-up`, `fetch-artifacts`, `replay-up`/`live-up`, `replay-down`/`live-down`/`down`,
`infra-down`):

```bash
make infra-up          # Redpanda + Redis (once per work session)
make fetch-artifacts   # idempotent - skips files already present locally
make replay-up         # replay_producer, inference_consumer, alert_panel, alert_dashboard
# or
make live-up           # chain_listener, inference_consumer, alert_panel, alert_dashboard
```

Then open `http://localhost:8501` for the alert dashboard (`http://localhost:8000/alerts` for
the raw backend JSON). `make down` stops the app stack only - infra is left running on purpose
(see "Two separate compose files" above); `make infra-down` stops that too.