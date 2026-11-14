# Minimal image for replay_producer.py (the "replay" event source - replays a historical sample,
# selected via the "replay" compose profile - alternative to chain_listener.py's "live" source,
# which has its own image.
#
# Directory structure under /app mirrors the repo (src/serving/...) - common.py
# computes PROJECT_ROOT as 2 parents above itself, so this must stay 2 levels
# deep for that math to still land on /app inside the container.
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "aiokafka>=0.14.0" \
    pandas \
    pyarrow \
    python-dotenv

COPY src/serving/common.py src/serving/replay_producer.py src/serving/

CMD ["python", "src/serving/replay_producer.py"]