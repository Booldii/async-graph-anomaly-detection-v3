# alert_dashboard.py's image - Streamlit frontend. --server.headless/--server.address are baked
# into CMD here so this image is correct by default

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit \
    streamlit-autorefresh \
    requests \
    pandas \
    python-dotenv

COPY src/serving/common.py src/serving/alert_dashboard.py src/serving/

CMD ["streamlit", "run", "src/serving/alert_dashboard.py", "--server.headless", "true", "--server.address", "0.0.0.0"]