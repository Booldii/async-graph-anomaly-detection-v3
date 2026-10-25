"""
Streamlit frontend for the Stage 3 alert panel - pure display layer over
src/serving/alert_panel.py's FastAPI backend.
"""

import time

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from common import ALERT_PANEL_URL

REFRESH_INTERVAL_MS = 5000
ALERTS_LIMIT = 200

# Thresholds only for the severity label column below - independent of inference_consumer.py's
# own alert threshold. These just flag among alerts already shown, which ones look most notable
# at a glance.
HIGH_ABSOLUTE_SCORE = 0.8
HIGH_SWING_DELTA = 0.4

st.set_page_config(page_title="TGN Risk Alerts", layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL_MS, key="alerts_refresh")

st.title("Stage 3 - Live Risk Alerts")

try:
    response = requests.get(f"{ALERT_PANEL_URL}/alerts", params={"n": ALERTS_LIMIT}, timeout=3)
    response.raise_for_status()
    alerts = response.json()
except requests.RequestException as e:
    st.error(f"Could not reach alert_panel.py backend at {ALERT_PANEL_URL}: {e}")
    st.stop()

st.subheader("Check a specific address")
query_address = st.text_input("Address", placeholder="0x...")
if query_address:
    try:
        score_response = requests.get(f"{ALERT_PANEL_URL}/score/{query_address.lower()}", timeout=3)
        if score_response.status_code == 404:
            st.warning(f"No score recorded yet for {query_address}")
        else:
            score_response.raise_for_status()
            st.metric("Current risk score", f"{score_response.json()['score']:.4f}")
    except requests.RequestException as e:
        st.error(f"Lookup failed: {e}")

st.divider()

if not alerts:
    st.info("No alerts yet.")
else:
    df = pd.DataFrame(alerts)

    alerts_last_hour = int((df["timestamp"] > time.time() - 3600).sum())
    st.metric("Alerts in the last hour", alerts_last_hour)

    def severity_label(row: pd.Series) -> str:
        # Independent, non-exclusive checks - an address can be both, shown as high + swing.
        high = row["score"] > HIGH_ABSOLUTE_SCORE
        swing = abs(row["delta"]) > HIGH_SWING_DELTA
        if high and swing:
            return "high + swing"
        if high:
            return "high"
        if swing:
            return "swing"
        return ""

    df["severity"] = df.apply(severity_label, axis=1)
    df["etherscan"] = df["address"].apply(lambda a: f"https://etherscan.io/address/{a}")
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")

    st.dataframe(
        df[["time", "severity", "address", "etherscan", "score", "prev_score", "delta"]],
        column_config={
            "time": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "severity": st.column_config.TextColumn("Severity"),
            "etherscan": st.column_config.LinkColumn("Etherscan", display_text="view"),
            "score": st.column_config.NumberColumn("Score", format="%.4f"),
            "prev_score": st.column_config.NumberColumn("Prev score", format="%.4f"),
            "delta": st.column_config.NumberColumn("Delta", format="%+.4f"),
        },
        hide_index=True,
    )