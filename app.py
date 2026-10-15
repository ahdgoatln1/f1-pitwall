import time
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="F1 Pitwall", layout="wide")
st.markdown(
    "<style>.stApp{background:#0b0d10;} "
    "html,body,[class*='css']{font-family:'Courier New',monospace;}</style>",
    unsafe_allow_html=True,
)

BASE = "https://api.openf1.org/v1"
TOP4 = ["McLaren", "Ferrari", "Red Bull", "Mercedes"]
KNOWN = ["Practice 1", "Practice 2", "Practice 3", "Qualifying",
         "Sprint Qualifying", "Sprint Shootout", "Sprint", "Race"]
COMPOUND_COLORS = {
    "SOFT": "#e8002d", "MEDIUM": "#ffd12e", "HARD": "#f0f0f0",
    "INTERMEDIATE": "#39b54a", "WET": "#1e6fd9",
}
BG = "#0b0d10"


@st.cache_data(ttl=1800, show_spinner=False)
def api(endpoint, params):
    q = dict(params)
    for attempt in range(5):
        r = requests.get(f"{BASE}/{endpoint}", params=q, timeout=60)
        if r.status_code == 200:
            time.sleep(0.4)
            data = r.json()
            return data if isinstance(data, list) else []
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError("OpenF1 rate limit reached. Wait a minute and retry.")


def frame(endpoint, **params):
    rows = api(endpoint, tuple(sorted(params.items())))
    return pd.DataFrame(rows)


def ensure(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fmt_lap(sec):
    if pd.isna(sec):
        return "—"
    m = int(sec // 60)
    return f"{m}:{sec - m * 60:06.3f}"


def fmt_sec(sec):
    return "—" if pd.isna(sec) else f"{sec:.3f}"


def year_sessions(year):
    df = frame("sessions", year=year)
    if df.empty:
        return df
    df = df[df["session_name"].isin(KNOWN)].copy()
    if df.empty:
        return df
    df["start"] = pd.to_datetime(df["date_start"], utc=True).dt.tz_localize(None)
    df["end"] = pd.to_datetime(df["date_end"], utc=True).dt.tz_localize(None)
    return df.sort_values("start").reset_index(drop=True)


now = pd.Timestamp.now(tz="UTC").tz_localize(None)

# ---------- header + live status ----------
st.markdown("### F1 PITWALL")
st.caption("Data: OpenF1 (free, historical). Sessions appear about 30 minutes after they end.")

try:
    cur = year_sessions(now.year)
    live = cur[(cur["start"] <= now) & (now < cur["end"])]
    nxt = cur[cur["start"] > now].sort_values("start")
except Exception as e:
    st.error(f"Could not reach OpenF1: {e}")
    st.stop()

if len(live) > 0:
    r0 = live.iloc[0]
    st.warning(
        f"SESSION IN PROGRESS: {r0['location']} - {r0['session_name']}. "
        "Free live telemetry is not available, so this session appears "
        "below once it has finished."
    )
else:
    msg = "NO LIVE SESSION"
    if len(nxt) > 0:
        n0 = nxt.iloc[0]
        msg += f" - next: {n0['location']} {n0['session_name']} ({n0['start']:%a %d %b %H:%M} UTC)"
    st.info(msg)

# ---------- session picker ----------
st.sidebar.header("Session")
year = st.sidebar.selectbox("Season", [2026, 2025, 2024])
try:
    sess = year_sessions(year)
except Exception as e:
    st.error(f"Could not load the calendar: {e}")
    st.stop()

if sess.empty:
    st.info("No sessions found for this season.")
    st.stop()

sess = sess[sess["end"] + timedelta(minutes=30) <= now]
if sess.empty:
    st.info("No finished sessions for this season yet.")
    st.stop()

meetings = (
    sess.groupby("meeting_key")
    .agg(first=("start", "min"), location=("location", "first"),
         country=("country_name", "first"))
    .sort_values("first")
)
options = []
for i, (mk, row) in enumerate(meetings.iterrows(), start=1):
    options.append((mk, f"R{i} - {row['location']} ({row['country']})"))
options.reverse()

mk_choice = st.sidebar.selectbox("Grand Prix", options, format_func=lambda x: x[1])
ms = sess[sess["meeting_key"] == mk_choice[0]].sort_values("start", ascending=False)
sess_choice = st.sidebar.selectbox(
    "Session", list(zip(ms["session_name"], ms["session_key"])),
    format_func=lambda x: x[0],
)
session_key = int(sess_choice[1])
top4_only = st.sidebar.checkbox("Top 4 teams only (McLaren, Ferrari, Red Bull, Mercedes)")

# ---------- load session data ----------
try:
    with st.spinner("Loading session data..."):
        drivers_df = frame("drivers", session_key=session_key)
        laps_df = frame("laps", session_key=session_key)
        stints_df = frame("stints", session_key=session_key)
        weather_df = frame("weather", session_key=session_key)
except Exception as e:
    st.error(f"Could not load this session: {e}")
    st.stop()

if drivers_df.empty or laps_df.empty:
    st.error("OpenF1 has no lap data for this session yet. Try again later or pick another session.")
    st.stop()

drivers_df = ensure(drivers_df, ["driver_number", "name_acronym", "full_name",
                                 "team_name", "team_colour"])
drivers_df = num(drivers_df, ["driver_number"]).drop_duplicates("driver_number")
laps_df = ensure(laps_df, ["driver_number", "lap_number", "lap_duration",
                           "duration_sector_1", "duration_sector_2",
                           "duration_sector_3", "is_pit_out_lap", "date_start"])
laps_df = num(laps_df, ["driver_number", "lap_number", "lap_duration",
                        "duration_sector_1", "duration_sector_2", "duration_sector_3"])
laps_df = laps_df.dropna(subset=["lap_number"])
stints_df = ensure(stints_df, ["driver_number", "stint_number", "lap_start",
                               "lap_end", "compound", "tyre_age_at_start"])
stints_df = num(stints_df, ["driver_number", "lap_start", "lap_end", "tyre_age_at_start"])

acr_of = dict(zip(drivers_df["driver_number"], drivers_df["name_acronym"]))
team_of = dict(zip(drivers_df["driver_number"], drivers_df["team_name"]))
col_of = dict(zip(drivers_df["driver_number"], drivers_df["team_colour"]))

view = drivers_df
if top4_only:
    mask = view["team_name"].astype(str).str.contains("|".join(TOP4), case=False, na=False)
    view = view[mask]
driver_nums = [int(n) for n in view["driver_number"].dropna()]
if not driver_nums:
    st.info("No drivers found for this selection.")
    st.stop()


def dlabel(n):
    return f"{acr_of.get(n, n)} - {team_of.get(n, '')}"


def dcolour(n):
    c = col_of.get(n)
    return f"#{c}" if isinstance(c, str) and len(c) == 6 else "#bbbbbb"


# ---------- weather ----------
if not weather_df.empty:
    weather_df = num(ensure(weather_df, ["air_temperature", "track_temperature",
                                         "humidity", "pressure", "wind_speed",
                                         "rainfall"]),
                     ["air_temperature", "track_temperature", "humidity",
                      "pressure", "wind_speed", "rainfall"])
    st.sidebar.header("Weather (session avg)")
    st.sidebar.write(f"Air: {weather_df['air_temperature'].mean():.1f} °C")
    st.sidebar.write(f"Track: {weather_df['track_temperature'].mean():.1f} °C")
    st.sidebar.write(f"Humidity: {weather_df['humidity'].mean():.0f} %")
    st.sidebar.write(f"Pressure: {weather_df['pressure'].mean():.1f} hPa")
    st.sidebar.write(f"Wind: {weather_df['wind_speed'].mean():.1f} m/s")
    st.sidebar.write("Rain: " + ("YES" if (weather_df["rainfall"] > 0).any() else "DRY"))
else:
    st.sidebar.caption("Weather data not available for this session.")

st.markdown(f"*{mk_choice[1]} {year} - {sess_choice[0]}*")
tab_pace, tab_tel, tab_tyre, tab_res = st.tabs(
    ["Pace monitor", "Telemetry + delta", "Tyre strategy", "Results"]
)

# ---------- pace ----------
with tab_pace:
    d = st.selectbox("Driver", driver_nums, format_func=dlabel, key="pace_d")
    ld = laps_df[laps_df["driver_number"] == d].sort_values("lap_number").copy()
    sd = stints_df[stints_df["driver_number"] == d]
    comps, ages = [], []
    for ln in ld["lap_number"]:
        c, a = None, None
        for _, s in sd.iterrows():
            ls, le = s["lap_start"], s["lap_end"]
            if pd.notna(ls) and ls <= ln and (pd.isna(le) or ln <= le):
                c = s["compound"]
                age0 = s["tyre_age_at_start"] if pd.notna(s["tyre_age_at_start"]) else 0
                a = age0 + (ln - ls)
                break
        comps.append(c)
        ages.append(a)
    ld["compound"] = comps
    ld["tyre_life"] = ages

    if len(ld) == 0:
        st.info("No laps recorded for this driver.")
    else:
        tbl = pd.DataFrame({
            "Lap": ld["lap_number"].astype(int),
            "Compound": ld["compound"],
            "LapTime": ld["lap_duration"].map(fmt_lap),
            "S1": ld["duration_sector_1"].map(fmt_sec),
            "S2": ld["duration_sector_2"].map(fmt_sec),
            "S3": ld["duration_sector_3"].map(fmt_sec),
            "TyreLife": ld["tyre_life"],
            "Note": ld["is_pit_out_lap"].map(
                lambda x: "OUT LAP" if pd.notna(x) and bool(x) else ""),
        })
        valid = ld.dropna(subset=["lap_duration"]).copy()
        if len(valid) > 0:
            hide = st.checkbox("Hide slow laps (over 107% of best)", value=True)
            if hide:
                valid = valid[valid["lap_duration"] <= valid["lap_duration"].min() * 1.07]
            fig = go.Figure(go.Scatter(
                x=valid["lap_number"], y=valid["lap_duration"], mode="lines+markers",
                line=dict(color="#888", width=1),
                marker=dict(size=8, color=[
                    COMPOUND_COLORS.get(str(c).upper(), "#888") for c in valid["compound"]]),
                text=valid["compound"],
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
@st.cache_data(ttl=3600, show_spinner=False)
def lap_telemetry(skey, dnum, start_iso, duration):
    start = pd.to_datetime(start_iso, utc=True).tz_localize(None)
    end = start + timedelta(seconds=float(duration))
    f = "%Y-%m-%dT%H:%M:%S.%f"
    params = {"session_key": int(skey), "driver_number": int(dnum),
              "date>": start.strftime(f), "date<": end.strftime(f)}
    return pd.DataFrame(api("car_data", tuple(sorted(params.items()))))


def prep_tel(df):
    if df.empty or len(df) < 20:
        return None
    df = df.copy()
    for c in ["speed", "throttle", "brake", "n_gear"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ts = pd.to_datetime(df["date"], utc=True)
    df["t"] = (ts - ts.min()).dt.total_seconds()
    df = df.sort_values("t").dropna(subset=["speed"])
    dt = df["t"].diff().fillna(0)
    df["dist"] = (df["speed"] / 3.6 * dt).cumsum()
    if df["dist"].max() <= 0:
        return None
    return df


def valid_laps(n):
    x = laps_df[(laps_df["driver_number"] == n)
                & laps_df["lap_duration"].notna()
                & laps_df["date_start"].notna()]
    return x.sort_values("lap_number")


def pick_lap(n, choice):
    x = valid_laps(n)
    if len(x) == 0:
        return None
    if choice == "Fastest":
        return x.loc[x["lap_duration"].idxmin()]
    row = x[x["lap_number"] == choice]
    return row.iloc[0] if len(row) else None


with tab_tel:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Driver A", driver_nums, index=0, format_func=dlabel, key="a")
    b = c2.selectbox("Driver B", driver_nums, index=min(1, len(driver_nums) - 1),
                     format_func=dlabel, key="b")
    la_opts = ["Fastest"] + [int(x) for x in valid_laps(a)["lap_number"]]
    lb_opts = ["Fastest"] + [int(x) for x in valid_laps(b)["lap_number"]]
    ch_a = c1.selectbox("Lap (A)", la_opts, key="la")
    ch_b = c2.selectbox("Lap (B)", lb_opts, key="lb")
    st.caption("Distance is estimated by integrating speed (OpenF1 has no distance "
               "channel), so the delta is approximate.")

    row_a, row_b = pick_lap(a, ch_a), pick_lap(b, ch_b)
    ta = tb = None
    if row_a is None or row_b is None:
        st.info("A selected driver has no timed lap.")
    else:
        try:
            ta = prep_tel(lap_telemetry(session_key, a, row_a["date_start"], row_a["lap_duration"]))
            tb = prep_tel(lap_telemetry(session_key, b, row_b["date_start"], row_b["lap_duration"]))
        except Exception as e:
            st.error(f"Could not load telemetry: {e}")
        if ta is None or tb is None:
            st.info("Telemetry is not available for this pair of laps.")
            ta = tb = None

    if ta is not None:
        m1, m2 = st.columns(2)
        diff = row_a["lap_duration"] - row_b["lap_duration"]
        m1.metric(f"{acr_of.get(a, a)} lap {int(row_a['lap_number'])}",
                  fmt_lap(row_a["lap_duration"]), f"{diff:+.3f}s vs {acr_of.get(b, b)}",
                  delta_color="inverse")
        m2.metric(f"{acr_of.get(b, b)} lap {int(row_b['lap_number'])}",
                  fmt_lap(row_b["lap_duration"]))

        length = (ta["dist"].max() + tb["dist"].max()) / 2
        grid = np.linspace(0, length, 700)

        def on_grid(t):
            d_ = t["dist"].to_numpy() / t["dist"].max() * length
            return {
                "speed": np.interp(grid, d_, t["speed"].to_numpy()),
                "throttle": np.interp(grid, d_, t["throttle"].to_numpy()),
                "brake": (np.interp(grid, d_, t["brake"].to_numpy()) > 50) * 100,
                "gear": np.rint(np.interp(grid, d_, t["n_gear"].to_numpy())),
            }

        ga, gb = on_grid(ta), on_grid(tb)
        delta = ga["speed"] - gb["speed"]
        col_a, col_b = dcolour(a), dcolour(b)
        if col_a == col_b:
            col_b = "#ffffff"
        na, nb = acr_of.get(a, str(a)), acr_of.get(b, str(b))

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.32, 0.14, 0.18, 0.12, 0.24])
        for g, col, nm in ((ga, col_a, na), (gb, col_b, nb)):
            fig.add_trace(go.Scatter(x=grid, y=g["speed"], name=nm, legendgroup=nm,
                                     line=dict(color=col, width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=grid, y=g["throttle"], name=nm, legendgroup=nm,
                                     showlegend=False, line=dict(color=col, width=1.5)),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=grid, y=g["brake"], name=nm, legendgroup=nm,
                                     showlegend=False,
                                     line=dict(color=col, width=1.5, shape="hv")),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=grid, y=g["gear"], name=nm, legendgroup=nm,
                                     showlegend=False,
                                     line=dict(color=col, width=1.5, shape="hv")),
                          row=5, col=1)
        fig.add_trace(go.Scatter(x=grid, y=delta, fill="tozeroy", showlegend=False,
                                 line=dict(color="#aaaaaa", width=1)), row=2, col=1)
        fig.update_yaxes(title_text="Speed km/h", row=1, col=1)
        fig.update_yaxes(title_text=f"Δ speed ({na}-{nb})", row=2, col=1)
        fig.update_yaxes(title_text="Throttle %", row=3, col=1)
        fig.update_yaxes(title_text="Brake", row=4, col=1)
        fig.update_yaxes(title_text="Gear", row=5, col=1)
        fig.update_xaxes(title_text="Track distance (m, estimated)", row=5, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                          height=950, hovermode="x unified",
                          legend=dict(orientation="h", y=1.04),
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------- tyres ----------
with tab_tyre:
    sv = stints_df[stints_df["driver_number"].isin(driver_nums)].copy()
    if len(sv) == 0:
        st.info("No tyre data for this session.")
    else:
        maxlap = laps_df.groupby("driver_number")["lap_number"].max()
        fig = go.Figure()
        for _, r in sv.iterrows():
            ls = r["lap_start"]
            le = r["lap_end"] if pd.notna(r["lap_end"]) else maxlap.get(r["driver_number"], ls)
            if pd.isna(ls):
                continue
            fig.add_trace(go.Bar(
                y=[acr_of.get(r["driver_number"], str(r["driver_number"]))],
                x=[le - ls + 1], base=[ls - 1], orientation="h", showlegend=False,
                marker=dict(color=COMPOUND_COLORS.get(str(r["compound"]).upper(), "#888"),
                            line=dict(color=BG, width=2)),
                hovertemplate=f"{r['compound']}: laps {int(ls)}-{int(le)}<extra></extra>",
            ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                          xaxis_title="Lap", barmode="overlay",
                          height=max(350, 28 * len(driver_nums) + 100),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red soft, yellow medium, white hard, green intermediate, blue wet.")

# ---------- results ----------
with tab_res:
    try:
        rs = frame("session_result", session_key=session_key)
    except Exception:
        rs = pd.DataFrame()
    if not rs.empty:
        rs = ensure(rs, ["position", "driver_number", "number_of_laps",
                         "gap_to_leader", "dnf", "dns", "dsq"])
        rs = num(rs, ["driver_number"])
        rs = rs[rs["driver_number"].isin(driver_nums)].sort_values("position")

        def status(r):
            if r["dsq"] is True:
                return "DSQ"
            if r["dns"] is True:
                return "DNS"
            if r["dnf"] is True:
                return "DNF"
            return ""

        out = pd.DataFrame({
            "Pos": rs["position"],
            "Driver": rs["driver_number"].map(acr_of),
            "Team": rs["driver_number"].map(team_of),
            "Laps": rs["number_of_laps"],
            "Gap": rs["gap_to_leader"].astype(str),
            "Status": rs.apply(status, axis=1),
        })
        st.subheader("Official result")
        st.dataframe(out, hide_index=True, use_container_width=True)
    else:
        st.caption("Official result not available for this session.")

    best = (laps_df.dropna(subset=["lap_duration"])
            .groupby("driver_number")["lap_duration"].min().sort_values())
    best = best[best.index.isin(driver_nums)]
    bt = pd.DataFrame({
        "Driver": [acr_of.get(n, n) for n in best.index],
        "Team": [team_of.get(n, "") for n in best.index],
        "Best lap": [fmt_lap(x) for x in best.values],
    })
    st.subheader("Best laps")
    st.dataframe(bt, hide_index=True, use_container_width=True)

# END OF FILE
