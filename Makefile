.PHONY: infra-up infra-down fetch-artifacts replay-up replay-down replay live-up live-down live down

INFRA_COMPOSE := docker/docker-compose.infra.yml
APP_COMPOSE := docker/docker-compose.app.yml

## One-shot: live risk scoring against a replayed historical sample (no on-chain connection needed).
replay: infra-up fetch-artifacts replay-up

## One-shot: live risk scoring against the real Ethereum mainnet feed (needs ALCHEMY_API_KEY in .env).
live: infra-up fetch-artifacts live-up

## Redpanda + Redis - started once per work session, left running across app rebuilds.
infra-up:
	docker compose -f $(INFRA_COMPOSE) up -d

infra-down:
	docker compose -f $(INFRA_COMPOSE) down

## Downloads the large model/data files the app containers bind-mount.
## Idempotent - skips files already present locally.
fetch-artifacts:
	uv run python scripts/fetch_artifacts.py

## App stack, replayed source: replay_producer, inference_consumer, alert_panel, alert_dashboard
replay-up:
	docker compose -f $(APP_COMPOSE) --profile replay up --build

replay-down:
	docker compose -f $(APP_COMPOSE) --profile replay down

## App stack, live source: chain_listener, inference_consumer, alert_panel, alert_dashboard
live-up:
	docker compose -f $(APP_COMPOSE) --profile live up --build

live-down:
	docker compose -f $(APP_COMPOSE) --profile live down

## Alias for replay-down - infra is left running
down: replay-down