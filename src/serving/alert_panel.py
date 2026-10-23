"""
Backend half of the Stage 3 alert panel.

GET /alerts reads the tgn:alerts list that inference_consumer.py's handle_score() pushes to on
every alert (LPUSH, newest first, capped at ALERTS_LIST_MAX_LEN there).

GET /score/{address} reads the same tgn:score:<address> key handle_score() always writes to
- lets the dashboard look up any address's current score on demand.
"""

import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from redis import asyncio as aioredis

from common import REDIS_URL

DEFAULT_ALERTS_LIMIT = 50
REDIS_ALERTS_KEY = "tgn:alerts"  # must match inference_consumer.py's REDIS_ALERTS_KEY
REDIS_SCORE_KEY_PREFIX = "tgn:score:"  # must match inference_consumer.py's REDIS_SCORE_KEY_PREFIX


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


@app.get("/score/{address}")
async def get_score(address: str):
    raw_score = await app.state.redis.get(REDIS_SCORE_KEY_PREFIX + address.lower())
    if raw_score is None:
        raise HTTPException(status_code=404, detail=f"no score recorded for {address}")
    return {"address": address.lower(), "score": float(raw_score)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)