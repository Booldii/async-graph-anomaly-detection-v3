# Minimal image for replay_producer.py (the demo event source). When chain_listener.py (live feed)\
# comes back into scope, it runs from this same image via a
# different "command:" + compose profile in docker-compose.app.yml
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