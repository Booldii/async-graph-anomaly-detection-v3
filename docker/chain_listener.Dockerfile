# Minimal image for chain_listener.py (the live, real Ethereum mainnet event source - alternative
# to replay_producer.py's historical replay, selected via the "live" compose profile).
#
# Directory structure under /app mirrors the repo (src/serving/...) - common.py
# computes PROJECT_ROOT as 2 parents above itself, so this must stay 2 levels
# deep for that math to still land on /app inside the container.
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "aiokafka>=0.14.0" \
    "web3>=8.0.0" \
    python-dotenv

COPY src/serving/common.py src/serving/chain_listener.py src/serving/

CMD ["python", "src/serving/chain_listener.py"]