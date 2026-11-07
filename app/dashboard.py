"""
VayuCast dashboard.

Run:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so "from src..." works when Streamlit runs this file

import altair as alt
import pandas as pd
import streamlit as st

from src.config import CITIES, PROCESSED_DIR, REPORTS_DIR
from src.predict import forecast_city

st.set_page_config(page_title="VayuCast — AQI forecasts", page_icon="🌬️", layout="wide")

# Official CPCB colours for each AQI category
COLORS = {"Good": "#00B050", "Satisfactory": "#92D050", "Moderate": "#E6C200",
          "Poor": "#FF9900", "Very Poor": "#FF0000", "Severe": "#C00000"}
ADVICE = {
    "Good": "Great day to be outside.",
    "Satisfactory": "Fine for most people. Very sensitive people take it easy.",
    "Moderate": "People with asthma or heart issues should limit long outdoor effort.",
    "Poor": "Avoid long outdoor exercise. Consider a mask if you're sensitive.",
    "Very Poor": "Stay indoors where possible. Wear an N95 outside.",
    "Severe": "Health alert: avoid going out. Keep windows closed.",
}


@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast(city: str) -> dict:
    return forecast_city(city)


@st.cache_data
def load_json(name: str):
    path = REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_daily() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "daily.csv", parse_dates=["date"])


def aqi_card(title: str, aqi: int, cat: str, sub: str = "") -> str:
    color = COLORS.get(cat, "#888")
    return f"""
    <div style="border-radius:14px;padding:16px 18px;background:{color}1A;
                border:1px solid {color}66;height:100%">
      <div style="font-size:0.85rem;opacity:0.75">{title}</div>
      <div style="font-size:2.4rem;font-weight:700;line-height:1.1">{aqi}</div>
      <div style="font-weight:600;color:{color}">{cat}</div>
      <div style="font-size:0.8rem;opacity:0.7;margin-top:4px">{sub}</div>
    </div>"""


# --------------------------------------------------------------------------- #
st.sidebar.title("🌬️ VayuCast")
st.sidebar.caption("Machine-learning air-quality forecasts for Indian cities")
city = st.sidebar.selectbox("City", list(CITIES), format_func=lambda c: CITIES[c]["name"])
metrics = load_json("metrics.json")
if metrics:
    st.sidebar.markdown(
        f"**Model:** {metrics['chosen_model']}  \n"
        f"**Trained:** {metrics['trained_at'][:10]}  \n"
        f"**1-day error:** ±{metrics['test']['1']['mae']:.0f} AQI on average")
st.sidebar.markdown("---")
st.sidebar.caption("Data: Open-Meteo (CAMS air quality + weather). AQI: CPCB method (PM2.5 & PM10). "
                   "Not an official forecast.")

tab_forecast, tab_all, tab_model, tab_data, tab_monitor = st.tabs(
    ["📍 Forecast", "🗺️ All cities", "📊 Model performance", "🔎 Data insights", "🩺 Monitoring"])

# ---- Forecast tab --------------------------------------------------------- #
with tab_forecast:
    try:
        with st.spinner("Fetching latest data and forecasting..."):
            fc = get_forecast(city)
    except Exception as error:
        st.error(f"Couldn't build a forecast right now ({error}). Try again in a minute.")
        st.stop()

    st.header(f"{fc['city_name']} — air quality forecast")
    last = fc["last_observed"]
    cols = st.columns(4)
    cols[0].markdown(aqi_card(f"Observed · {last['date']}", last["aqi"], last["category"],
                              f"PM2.5 {last['pm2_5']} · PM10 {last['pm10']} µg/m³"),
                     unsafe_allow_html=True)
    for col, f in zip(cols[1:], fc["forecast"]):
        label = pd.Timestamp(f["date"]).strftime("%a %d %b")
        col.markdown(aqi_card(f"Forecast · {label}", f["aqi"], f["category"],
                              f"likely range {f['low']}–{f['high']}"), unsafe_allow_html=True)

    tomorrow = fc["forecast"][0]
    st.info(f"**{tomorrow['category']}:** {ADVICE.get(tomorrow['category'], '')}")

    hist = pd.DataFrame(fc["history"]).assign(kind="observed")
    fut = pd.DataFrame(fc["forecast"]).assign(kind="forecast")
    hist["date"], fut["date"] = pd.to_datetime(hist["date"]), pd.to_datetime(fut["date"])
    bridge = pd.concat([hist.tail(1), fut])   # connect the observed line to the forecast
    chart = alt.layer(
        alt.Chart(fut).mark_area(opacity=0.2, color="#4c72b0").encode(
            x="date:T", y=alt.Y("low:Q", title="AQI"), y2="high:Q"),
        alt.Chart(hist).mark_line(color="#333", point=True).encode(
            x="date:T", y="aqi:Q", tooltip=["date:T", "aqi:Q"]),
        alt.Chart(bridge).mark_line(color="#4c72b0", strokeDash=[6, 4], point=True).encode(
            x="date:T", y="aqi:Q", tooltip=["date:T", "aqi:Q"]),
    ).properties(height=320, title="Last 30 days (black) and the next 3 days (blue, with likely range)")
    st.altair_chart(chart, use_container_width=True)

    if fc["drivers_tomorrow"]:
        st.subheader("Why this forecast? (top drivers for tomorrow)")
        st.caption("SHAP values: how many AQI points each factor pushed tomorrow's forecast "
                   "up (red) or down (green), compared with simply repeating today.")
        drv = pd.DataFrame(fc["drivers_tomorrow"])
        drv["direction"] = drv["effect"].apply(lambda e: "raises AQI" if e > 0 else "lowers AQI")
        st.altair_chart(
            alt.Chart(drv).mark_bar().encode(
                x=alt.X("effect:Q", title="Effect on forecast (AQI points)"),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color("direction:N", scale=alt.Scale(
                    domain=["raises AQI", "lowers AQI"], range=["#e45756", "#54a24b"]), legend=None),
                tooltip=["label", "value", "effect"]).properties(height=200),
            use_container_width=True)

# ---- All cities tab ------------------------------------------------------- #
with tab_all:
    st.header("Tomorrow across all cities")
    rows = []
    with st.spinner("Forecasting all cities..."):
        for cid in CITIES:
            try:
                f = get_forecast(cid)
                t = f["forecast"][0]
                rows.append({"City": f["city_name"], "Today (observed)": f["last_observed"]["aqi"],
                             "Tomorrow": t["aqi"], "Range": f"{t['low']}–{t['high']}",
                             "Category": t["category"]})
            except Exception as error:
                rows.append({"City": CITIES[cid]["name"], "Category": f"error: {error}"})
    table = pd.DataFrame(rows).sort_values("Tomorrow", ascending=False)
    st.dataframe(table.style.apply(
        lambda r: [f"background-color:{COLORS.get(r['Category'], '')}33"] * len(r), axis=1),
        hide_index=True, use_container_width=True)

# ---- Model performance tab ------------------------------------------------ #
with tab_model:
    st.header("How good is the model?")
    if not metrics:
        st.warning("No metrics yet — run `python -m src.train`.")
    else:
        st.markdown(f"Tested on **{metrics['test_period_start']} → {metrics['data_range'][1]}**, "
                    "a full year the model never saw while training.")
        c = st.columns(3)
        for col, h in zip(c, ["1", "2", "3"]):
            t = metrics["test"][h]
            col.metric(f"{h} day(s) ahead — average error", f"±{t['mae']:.1f} AQI",
                       f"{metrics['improvement_vs_persistence_pct'][h]:+.0f}% vs 'same as today'")
            col.caption(f"Right AQI category {t['category_accuracy']:.0%} of days · "
                        f"80% range covered {t['interval_coverage']:.0%}")
        comparison = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
        st.subheader("All models compared (test MAE, lower is better)")
        st.dataframe(comparison.pivot(index="model", columns="horizon", values="mae")
                     .round(1).sort_values(1), use_container_width=True)
        leak = metrics["leakage_demo"]
        st.subheader("The leakage trap")
        st.markdown(f"Testing with a **shuffled** split claims an error of **{leak['shuffled_kfold_mae']}**. "
                    f"Testing **honestly by time** gives **{leak['time_based_mae']}**. Shuffling leaks "
                    "future information, so it overstates how good the model is.")
        for img, caption in [("forecast_vs_actual.png", "Forecast vs actual on the test year"),
                             ("model_comparison.png", "Model comparison"),
                             ("shap_summary.png", "What drives the forecasts (SHAP)"),
                             ("error_by_city.png", "Error by city")]:
            path = REPORTS_DIR / "figures" / img
            if path.exists():
                st.image(str(path), caption=caption)

# ---- Data insights tab ---------------------------------------------------- #
with tab_data:
    st.header("What the data shows")
    daily = load_daily()
    daily["month"] = daily["date"].dt.month
    monthly = daily.groupby(["month", "city"])["aqi"].mean().reset_index()
    monthly["city"] = monthly["city"].map(lambda c: CITIES[c]["name"])
    st.altair_chart(alt.Chart(monthly).mark_line(point=True).encode(
        x=alt.X("month:O", title="Month"), y=alt.Y("aqi:Q", title="Average AQI"),
        color=alt.Color("city:N", title=None), tooltip=["city", "month", alt.Tooltip("aqi:Q", format=".0f")]
    ).properties(height=340, title="Seasonality: monsoon cleans the air, winter traps pollution"),
        use_container_width=True)
    pick = st.selectbox("Explore one city's full history", list(CITIES),
                        format_func=lambda c: CITIES[c]["name"], key="hist_city")
    one = daily[daily["city"] == pick]
    st.line_chart(one.set_index("date")["aqi"], height=260)
    st.caption(f"{len(daily):,} city-days of data · {daily['date'].min():%b %Y} – {daily['date'].max():%b %Y}")

# ---- Monitoring tab ------------------------------------------------------- #
with tab_monitor:
    st.header("Is the model still healthy?")
    mon = load_json("monitoring.json")
    if not mon:
        st.info("No monitoring report yet. It is created daily by `python -m src.monitor` "
                "(GitHub Actions runs it automatically).")
    else:
        st.markdown(f"Last check: **{mon['checked_at'][:16].replace('T', ' ')} UTC** — "
                    f"overall status: **{mon['status'].upper()}**")
        live = mon.get("live_performance") or {}
        if live.get("n_scored"):
            st.metric("Live error on real forecasts (1 day ahead)", f"±{live['mae_1day']:.1f} AQI",
                      f"{live['n_scored']} forecasts checked", delta_color="off")
        else:
            st.caption(live.get("note", ""))
        drift = pd.DataFrame(mon["drift"]).rename(columns={
            "feature": "Input", "shift_vs_normal_season": "Shift vs normal season (SD)",
            "status": "Status", "psi": "PSI", "psi_normal_max": "PSI seen in normal years (max)"})
        st.dataframe(drift, hide_index=True, use_container_width=True)
        st.caption("Each input over the last 30 days, compared with the same city and season in previous "
                   "years. ±1 SD is normal, 1–2 is unusual, above 2 is extreme. PSI is shown for reference "
                   "only: on 30-day windows of daily data it is noisy even in normal years.")
