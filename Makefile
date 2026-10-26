.PHONY: infra-up infra-down fetch-artifacts demo-up demo-down demo down

INFRA_COMPOSE := docker/docker-compose.infra.yml
APP_COMPOSE := docker/docker-compose.app.yml

## One-shot: everything needed to see the demo running, in the right order.
demo: infra-up fetch-artifacts demo-up

## Redpanda + Redis - started once per work session, left running across app rebuilds.
infra-up:
	docker compose -f $(INFRA_COMPOSE) up -d

infra-down:
	docker compose -f $(INFRA_COMPOSE) down

## Downloads the large model/data files the app containers bind-mount.
## Idempotent - skips files already present locally.
fetch-artifacts:
	uv run python scripts/fetch_artifacts.py

## App stack: replay_producer, inference_consumer, alert_panel, alert_dashboard
demo-up:
	docker compose -f $(APP_COMPOSE) --profile demo up --build

demo-down:
	docker compose -f $(APP_COMPOSE) --profile demo down

## Alias for demo-down - infra is left running
down: demo-down