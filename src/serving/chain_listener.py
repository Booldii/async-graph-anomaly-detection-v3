"""
Connects to Ethereum Mainnet over a persistent WebSocket (Alchemy) and subscribes to all
ERC-20 Transfer logs network-wide. Decodes and normalizes each log into the same row shape produced
by src/data/bq_fetch_day.py, then publishes it downstream to the Redpanda broker started by
docker-compose.infra.yml

Known MVP limitation: the reconnect/backfill logic below only covers gaps from a WebSocket
drop within one process lifetime. last_processed_block is in-memory only - a full process
restart starts fresh with no backfill.
"""

import asyncio
import json
import logging
import os

from aiokafka import AIOKafkaProducer
from common import KAFKA_BOOTSTRAP_SERVERS, TOPIC_TRANSFERS
from eth_typing import ChecksumAddress
from web3 import AsyncWeb3, WebSocketProvider
from web3.exceptions import TimeExhausted
from web3.types import FilterParams, LogReceipt

ALCHEMY_API_KEY = os.environ["ALCHEMY_API_KEY"]
WS_URL = f"wss://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

# Signature-only hash - identical for ERC-20's Transfer(address,address,uint256) and ERC-721's
# Transfer(address,address,uint256) (tokenId), since keccak only depends on the text.
# _normalize_log tells them apart via topics length.
TRANSFER_EVENT_TOPIC = AsyncWeb3.keccak(text="Transfer(address,address,uint256)").to_0x_hex()

MINIMAL_ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    }
]
# Matches src/data/bq_fetch_day.py's fallback for tokens with no on-chain decimals() or an
# unrecognized contract
DEFAULT_DECIMALS_FALLBACK = 18

RECONNECT_BACKOFF_SECONDS = 5

# Alchemy's free tier rejects eth_getLogs for ranges wider than this  - a PAYG plan
# lifts it, but backfill() chunks regardless so it degrades gracefully either way.
MAX_LOG_RANGE_BLOCKS = 10

# websockets` own default (1 MiB) is too small for a network-wide eth_getLogs response or an
# unusually busy block's push notification -
MAX_WS_FRAME_BYTES = 20 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chain_listener")


class TokenDecimalsCache:
    """Per-contract decimals are effectively permanent, and the universe of distinct tokens
    seen stays small relative to event volume - so this cache is never evicted and is kept
    across reconnects."""

    def __init__(self) -> None:
        self._cache: dict[str, int] = {}

    async def get(self, w3: AsyncWeb3, contract_address: ChecksumAddress) -> int:
        if contract_address not in self._cache:
            contract = w3.eth.contract(address=contract_address, abi=MINIMAL_ERC20_ABI)
            try:
                self._cache[contract_address] = await contract.functions.decimals().call()
            except Exception:
                logger.warning(
                    "decimals() call failed for %s, falling back to %d",
                    contract_address,
                    DEFAULT_DECIMALS_FALLBACK,
                )
                self._cache[contract_address] = DEFAULT_DECIMALS_FALLBACK
        return self._cache[contract_address]


class BoundedFifoCache:
    """FIFO-evicted cache for block timestamps / tx gas info - only needed transiently (the
    current block's logs, or a batch-mint tx's many logs sharing one receipt), unlike token
    decimals which are a small, effectively permanent universe."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: dict = {}

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        if key not in self._data and len(self._data) >= self._maxsize:
            del self._data[next(iter(self._data))]
        self._data[key] = value


async def normalize_log(
    w3: AsyncWeb3,
    log: LogReceipt,
    decimals_cache: TokenDecimalsCache,
    block_ts_cache: BoundedFifoCache,
    gas_cache: BoundedFifoCache,
) -> dict | None:
    topics = log["topics"]
    if len(topics) != 3:
        # 4 topics = ERC-721 Transfer(from, to, indexed tokenId) - same signature hash as ERC-20
        # <3 --> malformed/non-standard, also skipped.
        return None

    from_addr = AsyncWeb3.to_checksum_address(bytes(topics[1])[-20:])
    to_addr = AsyncWeb3.to_checksum_address(bytes(topics[2])[-20:])
    raw_value = int.from_bytes(log["data"], "big")

    contract_address = log["address"]
    decimals = await decimals_cache.get(w3, contract_address)

    block_number = log["blockNumber"]
    if block_number not in block_ts_cache:
        block = await w3.eth.get_block(block_number)
        block_ts_cache[block_number] = block["timestamp"]
    timestamp = block_ts_cache[block_number]

    tx_hash = log["transactionHash"]
    if tx_hash not in gas_cache:
        receipt = await w3.eth.get_transaction_receipt(tx_hash)
        gas_cache[tx_hash] = (receipt["gasUsed"], receipt["effectiveGasPrice"])
    gas_used, gas_price = gas_cache[tx_hash]

    return {
        "hash": tx_hash.to_0x_hex(),
        "logIndex": log["logIndex"],
        "blockNumber": block_number,
        "from": from_addr,
        "to": to_addr,
        "value": raw_value / (10**decimals),
        "timeStamp": timestamp,
        "contractAddress": contract_address,
        "tokenDecimal": decimals,
        "gasPrice": gas_price,
        "gasUsed": gas_used,
    }


async def publish(producer: AIOKafkaProducer, row: dict) -> None:
    # The broker is Redpanda,  which speaks the Kafka wire protocol,
    # so any standard Kafka client library works against it unmodified.
    await producer.send_and_wait(TOPIC_TRANSFERS, json.dumps(row).encode("utf-8"))


async def backfill(
    w3: AsyncWeb3,
    from_block: int,
    decimals_cache: TokenDecimalsCache,
    block_ts_cache: BoundedFifoCache,
    gas_cache: BoundedFifoCache,
    producer: AIOKafkaProducer,
) -> None:
    to_block = await w3.eth.block_number
    if from_block > to_block:
        return
    total_logs = 0
    # Chunked into MAX_LOG_RANGE_BLOCKS-sized windows - a single call for the whole gap would
    # exceed Alchemy's free-tier eth_getLogs range limit as soon as the gap passes 10 blocks.
    chunk_start = from_block
    while chunk_start <= to_block:
        chunk_end = min(chunk_start + MAX_LOG_RANGE_BLOCKS - 1, to_block)
        filter_params: FilterParams = {
            "fromBlock": chunk_start,
            "toBlock": chunk_end,
            "topics": [TRANSFER_EVENT_TOPIC],
        }
        logs = await w3.eth.get_logs(filter_params)
        for log in logs:
            row = await normalize_log(w3, log, decimals_cache, block_ts_cache, gas_cache)
            if row is not None:
                await publish(producer, row)
        total_logs += len(logs)
        chunk_start = chunk_end + 1
    logger.info("backfill done: %d logs replayed", total_logs)

def _quiet_queue_full_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """asyncio's default behavior for an exception raised inside a task's done-callback is
    to dump a raw traceback via its own exception handler. web3.py's internal subscription queue
    overflowing under full mainnet volume raises exactly this - asyncio.QueueFull - repeatedly while
    we're already reconnecting."""
    exc = context.get("exception")
    if isinstance(exc, asyncio.QueueFull):
        logger.warning("live feed is falling behind full mainnet volume (internal queue overflow)")
        return
    loop.default_exception_handler(context)


async def run() -> None:
    asyncio.get_running_loop().set_exception_handler(_quiet_queue_full_handler)

    decimals_cache = TokenDecimalsCache()
    block_ts_cache = BoundedFifoCache(maxsize=256)
    gas_cache = BoundedFifoCache(maxsize=4096)
    last_processed_block: int | None = None

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await producer.start()
    try:
        while True:
            try:
                async with AsyncWeb3(
                    WebSocketProvider(WS_URL, websocket_kwargs={"max_size": MAX_WS_FRAME_BYTES})
                ) as w3:

                    async def handler(ctx) -> None:
                        nonlocal last_processed_block
                        last_processed_block = ctx.result["blockNumber"]
                        row = await normalize_log(
                            ctx.async_w3, ctx.result, decimals_cache, block_ts_cache, gas_cache
                        )
                        if row is not None:
                            await publish(producer, row)

                    # Subscribe before backfilling: any log for a block mined while backfill()
                    # runs is still queued by the provider and handled once we start draining
                    # below, so nothing is missed. The backfill range can overlap by a block or
                    # two with what the live subscription also delivers - tolerated as harmless
                    # duplicates rather than engineered away
                    await w3.eth.subscribe("logs", {"topics": [TRANSFER_EVENT_TOPIC]}, handler=handler)
                    logger.info("subscribed to Transfer logs")

                    if last_processed_block is not None:
                        try:
                            await backfill(
                                w3, last_processed_block, decimals_cache, block_ts_cache, gas_cache, producer
                            )
                        except (TimeExhausted, asyncio.QueueFull):
                            # Same known overflow symptom as the outer except below
                            logger.warning(
                                "backfill skipped - live feed is falling behind full mainnet volume"
                            )
                        except Exception:
                            # A backfill failure must not tear down an otherwise healthy subscription
                            # by falling through to the outer except below - log and keep going. Any logs
                            # missed here are a small, bounded gap.
                            logger.exception("backfill failed, continuing without it")

                    await w3.subscription_manager.handle_subscriptions(run_forever=True)
            except (TimeExhausted, asyncio.QueueFull):
                # The other half of the overflow symptom _quiet_queue_full_handler quiets above -
                # this is the path where the queue backup surfaces as a stalled RPC response or
                # a directly-propagated QueueFull.
                logger.warning(
                    "live feed is falling behind full mainnet volume - reconnecting in %ds...",
                    RECONNECT_BACKOFF_SECONDS,
                )
                await asyncio.sleep(RECONNECT_BACKOFF_SECONDS)
            except Exception:
                logger.exception("connection lost, reconnecting in %ds...", RECONNECT_BACKOFF_SECONDS)
                await asyncio.sleep(RECONNECT_BACKOFF_SECONDS)
    finally:
        await producer.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("stopped")