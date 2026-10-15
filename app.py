import os
from datetime import timedelta

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="F1 Pitwall", layout="wide")
st.markdown(
    "<style>.stApp{background:#0b0d10;} "
    "html,body,[class*='css']{font-family:'Courier New',monospace;}</style>",
    unsafe_allow_html=True,
)

TOP4 = ["McLaren", "Ferrari", "Red Bull", "Mercedes"]
COMPOUND_COLORS = {
    "SOFT": "#e8002d", "MEDIUM": "#ffd12e", "HARD": "#f0f0f0",
    "INTERMEDIATE": "#39b54a", "WET": "#1e6fd9", "UNKNOWN": "#888888",
}
BG = "#0b0d10"


@st.cache_resource
def setup_cache():
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")
    return True


setup_cache()


def fmt_lap(td):
    if pd.isna(td):
        return "—"
    t = td.total_seconds()
    m = int(t // 60)
    return f"{m}:{t - m * 60:06.3f}"


def fmt_sec(td):
    if pd.isna(td):
        return "—"
    return f"{td.total_seconds():.3f}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule(year):
    return fastf1.get_event_schedule(year, include_testing=False)


def sessions_of(ev):
    out = []
    for i in range(1, 6):
        name = ev.get(f"Session{i}")
        start = ev.get(f"Session{i}DateUtc")
        if name is None or pd.isna(name) or str(name).strip() in ("", "None"):
            continue
        if start is None or pd.isna(start):
            continue
        out.append((str(name), pd.Timestamp(start)))
    return out


def end_of(name, start):
    return start + timedelta(hours=3 if name == "Race" else 1.5)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_session(year, rnd, name):
    s = fastf1.get_session(year, rnd, name)
    s.load(laps=True, telemetry=True, weather=True, messages=False)
    return s


now = pd.Timestamp.now(tz="UTC").tz_localize(None)

# ---------- header + live status ----------
st.markdown("### F1 PITWALL")

live, nxt = None, None
try:
    cur = load_schedule(now.year)
    for _, ev in cur.iterrows():
        for name, start in sessions_of(ev):
            if start <= now < end_of(name, start):
                live = (ev["EventName"], name)
            elif start > now and (nxt is None or start < nxt[2]):
                nxt = (ev["EventName"], name, start)
except Exception as e:
    st.warning(f"Could not read the live calendar: {e}")

if live:
    st.warning(
        f"SESSION WINDOW OPEN: {live[0]} - {live[1]}. Free live telemetry "
        "is not available, so full data appears below once the session is over."
    )
else:
    msg = "NO LIVE SESSION"
    if nxt:
        msg += f" - next: {nxt[0]} {nxt[1]} ({nxt[2]:%a %d %b %H:%M} UTC)"
    st.info(msg)

# ---------- session picker ----------
st.sidebar.header("Session")
year = st.sidebar.selectbox("Season", [2026, 2025, 2024])
sched = load_schedule(year)

available = []
for _, ev in sched.iterrows():
    done = [n for n, s in sessions_of(ev) if end_of(n, s) <= now]
    if done:
        available.append((int(ev["RoundNumber"]), ev["EventName"], done))
available.reverse()  # newest first

if not available:
    st.info("No finished sessions found for this season yet.")
    st.stop()

ev_choice = st.sidebar.selectbox(
    "Grand Prix", available, format_func=lambda x: f"R{x[0]} - {x[1]}"
)
sess_name = st.sidebar.selectbox("Session", ev_choice[2][::-1])
top4_only = st.sidebar.checkbox("Top 4 teams only (McLaren, Ferrari, Red Bull, Mercedes)")

try:
    with st.spinner("Loading session data (first load can take a minute or two)..."):
        session = load_session(year, ev_choice[0], sess_name)
except Exception as e:
    st.error(
        "Could not load this session. Data can appear some time after a "
        f"session ends. Details: {e}"
    )
    st.stop()

# ---------- drivers ----------
res = session.results
if res is not None and len(res) > 0:
    if top4_only:
        mask = res["TeamName"].str.contains("|".join(TOP4), case=False, na=False)
        res_view = res[mask]
    else:
        res_view = res
    drivers = list(res_view["Abbreviation"])
    team_of = dict(zip(res["Abbreviation"], res["TeamName"]))
else:
    drivers = sorted(session.laps["Driver"].unique())
    team_of = {}

if not drivers:
    st.info("No drivers found for this selection.")
    st.stop()


def dlabel(d):
    return f"{d} - {team_of[d]}" if d in team_of else d


# ---------- weather ----------
wd = session.weather_data
if wd is not None and len(wd) > 0:
    st.sidebar.header("Weather (session avg)")
    st.sidebar.write(f"Air: {wd['AirTemp'].mean():.1f} °C")
    st.sidebar.write(f"Track: {wd['TrackTemp'].mean():.1f} °C")
    st.sidebar.write(f"Humidity: {wd['Humidity'].mean():.0f} %")
    st.sidebar.write(f"Pressure: {wd['Pressure'].mean():.1f} hPa")
    st.sidebar.write(f"Wind: {wd['WindSpeed'].mean():.1f} m/s")
    st.sidebar.write("Rain: " + ("YES" if wd["Rain"].any() else "DRY"))

st.markdown(f"*{session.event['EventName']} {year} - {sess_name}*")
tab_pace, tab_tel, tab_tyre, tab_res = st.tabs(
    ["Pace monitor", "Telemetry + delta", "Tyre strategy", "Results"]
)

# ---------- pace ----------
with tab_pace:
    d = st.selectbox("Driver", drivers, format_func=dlabel, key="pace_d")
    laps = session.laps.pick_drivers(d).copy()
    if len(laps) == 0:
        st.info("No laps recorded for this driver.")
    else:
        tbl = pd.DataFrame({
            "Lap": laps["LapNumber"].astype(int),
            "Compound": laps["Compound"],
            "LapTime": laps["LapTime"].map(fmt_lap),
            "S1": laps["Sector1Time"].map(fmt_sec),
            "S2": laps["Sector2Time"].map(fmt_sec),
            "S3": laps["Sector3Time"].map(fmt_sec),
            "TyreLife": laps["TyreLife"],
        })
        valid = laps.dropna(subset=["LapTime"]).copy()
        if len(valid) > 0:
            valid["sec"] = valid["LapTime"].dt.total_seconds()
            hide = st.checkbox("Hide slow laps (over 107% of best)", value=True)
            if hide:
                valid = valid[valid["sec"] <= valid["sec"].min() * 1.07]
            fig = go.Figure(go.Scatter(
                x=valid["LapNumber"], y=valid["sec"], mode="lines+markers",
                line=dict(color="#888", width=1),
                marker=dict(
                    size=8,
                    color=[COMPOUND_COLORS.get(c, "#888") for c in valid["Compound"]],
                ),
                text=valid["Compound"],
                hovertemplate="Lap %{x}<br>%{y:.3f}s<br>%{text}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                xaxis_title="Lap", yaxis_title="Lap time (s)", height=380,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tbl, hide_index=True, use_container_width=True)

# ---------- telemetry ----------
with tab_tel:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Driver A", drivers, index=0, format_func=dlabel, key="a")
    b = c2.selectbox(
        "Driver B", drivers, index=min(1, len(drivers) - 1),
        format_func=dlabel, key="b",
    )
    st.caption("Compares each driver's fastest lap of this session.")
    try:
        lap_a = session.laps.pick_drivers(a).pick_fastest()
        lap_b = session.laps.pick_drivers(b).pick_fastest()
        if lap_a is None or lap_b is None:
            raise ValueError("a selected driver has no timed lap")
        tel_a = lap_a.get_car_data().add_distance()
        tel_b = lap_b.get_car_data().add_distance()
    except Exception as e:
        st.info(f"Telemetry not available for this pair: {e}")
        tel_a = None

    if tel_a is not None:
        m1, m2 = st.columns(2)
        diff = (lap_a["LapTime"] - lap_b["LapTime"]).total_seconds()
        m1.metric(f"{a} fastest", fmt_lap(lap_a["LapTime"]), f"{diff:+.3f}s vs {b}", delta_color="inverse")
        m2.metric(f"{b} fastest", fmt_lap(lap_b["LapTime"]))

        speed_b = np.interp(tel_a["Distance"], tel_b["Distance"], tel_b["Speed"])
        delta = tel_a["Speed"].to_numpy() - speed_b
        col_a, col_b = "#ff8fb1", "#5ce1e6"

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[0.32, 0.14, 0.18, 0.12, 0.24],
        )
        for tel, col, nm in ((tel_a, col_a, a), (tel_b, col_b, b)):
            fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["Speed"], name=nm,
                                     line=dict(color=col, width=1.5), legendgroup=nm), row=1, col=1)
            fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["Throttle"], name=nm,
                                     line=dict(color=col, width=1.5), legendgroup=nm,
                                     showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["Brake"].astype(int) * 100,
                                     name=nm, line=dict(color=col, width=1.5, shape="hv"),
                                     legendgroup=nm, showlegend=False), row=4, col=1)
            fig.add_trace(go.Scatter(x=tel["Distance"], y=tel["nGear"], name=nm,
                                     line=dict(color=col, width=1.5, shape="hv"),
                                     legendgroup=nm, showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=tel_a["Distance"], y=delta, name=f"{a} - {b}",
                                 fill="tozeroy", line=dict(color="#aaaaaa", width=1),
                                 showlegend=False), row=2, col=1)
        fig.update_yaxes(title_text="Speed km/h", row=1, col=1)
        fig.update_yaxes(title_text=f"Δ speed ({a}-{b})", row=2, col=1)
        fig.update_yaxes(title_text="Throttle %", row=3, col=1)
        fig.update_yaxes(title_text="Brake", row=4, col=1)
        fig.update_yaxes(title_text="Gear", row=5, col=1)
        fig.update_xaxes(title_text="Track distance (m)", row=5, col=1)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            height=950, hovermode="x unified",
            legend=dict(orientation="h", y=1.04),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- tyres ----------
with tab_tyre:
    all_laps = session.laps
    all_laps = all_laps[all_laps["Driver"].isin(drivers)]
    stints = (
        all_laps.groupby(["Driver", "Stint", "Compound"])
        .agg(start=("LapNumber", "min"), end=("LapNumber", "max"))
        .reset_index()
    )
    if len(stints) == 0:
        st.info("No tyre data for this session.")
    else:
        fig = go.Figure()
        for _, r in stints.iterrows():
            fig.add_trace(go.Bar(
                y=[r["Driver"]], x=[r["end"] - r["start"] + 1], base=[r["start"] - 1],
                orientation="h", showlegend=False,
                marker=dict(color=COMPOUND_COLORS.get(r["Compound"], "#888"),
                            line=dict(color=BG, width=2)),
                hovertemplate=f"{r['Compound']}: laps {int(r['start'])}-{int(r['end'])}<extra></extra>",
            ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis_title="Lap", barmode="overlay",
            height=max(350, 28 * len(drivers) + 100),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red soft, yellow medium, white hard, green intermediate, blue wet.")

# ---------- results ----------
with tab_res:
    if res is None or len(res) == 0:
        st.info("No results available.")
    else:
        cols = [c for c in ["Position", "Abbreviation", "FullName", "TeamName",
                            "GridPosition", "Status", "Points"] if c in res_view.columns]
        st.dataframe(res_view[cols], hide_index=True, use_container_width=True)
