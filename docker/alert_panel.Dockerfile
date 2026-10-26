# alert_panel.py's image - FastAPI backend for the alert dashboard. Already binds
# host="0.0.0.0" in its own uvicorn.run() call, so no extra flags needed
# here the way alert_dashboard.alert_dashboard.inference_consumer.Dockerfile needs for Streamlit.
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    redis \
    python-dotenv

COPY src/serving/common.py src/serving/alert_panel.py src/serving/

CMD ["python", "src/serving/alert_panel.py"]