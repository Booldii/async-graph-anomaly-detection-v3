"""
Shared constants for the event-driven services - one definition of the broker topic and
default Redpanda/Redis addresses, so all services can't drift out of agreement with each
other or with docker.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Matches the topic-init service in docker-compose.infra.yml using 1 partition,
# since per-address TGN memory updates must stay in on-chain arrival order.
TOPIC_TRANSFERS = "eth-transfers"

# Defaults match docker-compose.infra.yml's host-exposed ports (Redpanda's external listener,
# Redis' default port) - overridable via env for anything other than this local demo stack.
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")