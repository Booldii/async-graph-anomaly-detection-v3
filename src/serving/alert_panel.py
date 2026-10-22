"""
Backend half of the Stage 3 alert panel. Reads the tgn:alerts list that inference_consumer.py's handle_score()
pushes to on every alert (LPUSH, newest first, capped at ALERTS_LIST_MAX_LEN there) and serves it as JSON.
"""

import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Query
from redis import asyncio as aioredis

from common import REDIS_URL

DEFAULT_ALERTS_LIMIT = 50
REDIS_ALERTS_KEY = "tgn:alerts"  # matches inference_consumer.py's REDIS_ALERTS_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        yield
    finally:
        await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/alerts")
async def get_alerts(n: int = Query(default=DEFAULT_ALERTS_LIMIT, ge=1, le=1000)):
    raw_records = await app.state.redis.lrange(REDIS_ALERTS_KEY, 0, n - 1)
    return [json.loads(record) for record in raw_records]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)