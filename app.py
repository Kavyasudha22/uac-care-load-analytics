"""
System Capacity & Care Load Analytics for Unaccompanied Children  -  Streamlit dashboard
Run locally:   streamlit run app.py
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. CONSTANTS
# ----------------------------------------------------------------------
RENAME = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "apprehended",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transferred",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharged",
}
STOCKS = ["cbp_custody", "hhs_care"]                      # "how many are in care today"
FLOWS = ["apprehended", "transferred", "discharged"]      # "how many moved today"

LABELS = {
    "cbp_custody": "Children in CBP custody",
    "hhs_care": "Children in HHS care",
    "total_load": "Total system load (CBP + HHS)",
    "apprehended": "Apprehended (intake)",
    "transferred": "Transferred CBP -> HHS",
    "discharged": "Discharged from HHS",
    "net_intake": "Net daily intake (transfers - discharges)",
    "growth_rate": "Care load growth rate (%)",
    "load_ma7": "Total load, 7-day avg",
    "load_ma14": "Total load, 14-day avg",
    "net_ma7": "Net intake, 7-day avg",
    "net_ma14": "Net intake, 14-day avg",
    "cum_net_intake": "Cumulative net intake (est.)",
}


# ----------------------------------------------------------------------
# 2. INGESTION
# ----------------------------------------------------------------------
def load_raw(src):
    """Read the HHS CSV and return (clean_dataframe, info_dict).

    Handles: blank trailing rows, the '*' in a column name, thousands
    separators in 'Children in HHS Care' ("2,484"), and 'Month DD, YYYY' dates.
    """
    df = pd.read_csv(src)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    missing_cols = set(RENAME.values()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing_cols)}")
    df = df[list(RENAME.values())]

    n_total = len(df)
    df = df.dropna(how="all")
    n_blank = n_total - len(df)

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    for c in STOCKS + FLOWS:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    n_bad_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    info = {"rows_in_file": n_total, "blank_rows_dropped": n_blank,
            "unparseable_dates_dropped": n_bad_dates}
    return df, info


def build_daily(raw):
    """Sort, de-duplicate and reindex to a COMPLETE daily calendar.

    * Stock columns (census counts) are linearly interpolated across gaps.
    * Flow columns (daily movements) are left as NaN on unreported days --
      we never invent movements. `reported` marks real observations.
    """
    n_dups = int(raw["date"].duplicated().sum())
    df = raw.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    d = df.reindex(full_idx)
    d.index.name = "date"
    d["reported"] = d.index.isin(df.index)
    d[STOCKS] = d[STOCKS].interpolate(method="time", limit_area="inside")
    d.attrs["duplicates_removed"] = n_dups
    return d


# ----------------------------------------------------------------------
# 3. VALIDATION
# ----------------------------------------------------------------------
def add_validation_flags(d):
    d = d.copy()
    d["flag_transfer_gt_cbp"] = d["transferred"] > d["cbp_custody"]
    d["flag_discharge_gt_hhs"] = d["discharged"] > d["hhs_care"]
    d["flag_negative"] = (d[STOCKS + FLOWS] < 0).any(axis=1)
    return d


def validation_summary(d, info=None):
    rep = d[d["reported"]]
    s = {
        "Date range": f"{d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d}",
        "Calendar days in range": len(d),
        "Days with a report": int(d["reported"].sum()),
        "Missing calendar days": int((~d["reported"]).sum()),
        "Reporting coverage (%)": round(100 * d["reported"].mean(), 1),
        "Duplicate dates removed": d.attrs.get("duplicates_removed", 0),
        "Transfers > CBP custody (days)": int(d["flag_transfer_gt_cbp"].sum()),
        "Discharges > HHS care (days)": int(d["flag_discharge_gt_hhs"].sum()),
        "Negative values (days)": int(d["flag_negative"].sum()),
        "Zero-intake reported days": int((rep["apprehended"] == 0).sum()),
    }
    if info:
        s["Blank rows dropped from file"] = info["blank_rows_dropped"]
    return pd.Series(s, name="value")


def reconciliation(d):
    """Does the change in HHS census match (transfers - discharges)?
    Uses only consecutive reported days."""
    r = d[d["reported"]].copy()
    gap = r.index.to_series().diff().dt.days
    r["census_change"] = r["hhs_care"].diff()
    r["net_flow"] = r["transferred"] - r["discharged"]
    r = r[gap == 1].dropna(subset=["census_change", "net_flow"])
    return {
        "pairs_compared": len(r),
        "correlation": float(r["census_change"].corr(r["net_flow"])),
        "mean_census_change": float(r["census_change"].mean()),
        "mean_net_flow": float(r["net_flow"].mean()),
    }


# ----------------------------------------------------------------------
# 4. DERIVED CAPACITY METRICS
# ----------------------------------------------------------------------
def run_len(mask):
    """For each True element, the total length of the consecutive-True run it
    belongs to (0 for False elements)."""
    mask = mask.fillna(False).astype(bool)
    grp = mask.ne(mask.shift(fill_value=False)).cumsum()
    return mask.astype(int).groupby(grp).transform("sum") * mask.astype(int)


def compute_metrics(d, strain_pct=75, min_strain_days=14, min_backlog_days=7):
    d = d.copy()

    # --- core metrics -------------------------------------------------
    d["total_load"] = d["cbp_custody"] + d["hhs_care"]
    d["net_intake"] = d["transferred"] - d["discharged"]
    d["growth_rate"] = d["total_load"].pct_change() * 100
    # estimate for gap days, used ONLY for the cumulative curve
    d["net_intake_est"] = (
        d["net_intake"].interpolate(method="time", limit_area="inside").fillna(0)
    )
    d["cum_net_intake"] = d["net_intake_est"].cumsum()

    # --- rolling windows ----------------------------------------------
    for w in (7, 14):
        d[f"load_ma{w}"] = d["total_load"].rolling(w, min_periods=1).mean()
        d[f"load_std{w}"] = d["total_load"].rolling(w, min_periods=2).std()
        d[f"net_ma{w}"] = d["net_intake"].rolling(w, min_periods=2).mean()
    d["load_cv14"] = 100 * d["load_std14"] / d["load_ma14"]
    d["growth_std14"] = d["growth_rate"].rolling(14, min_periods=5).std()

    # --- backlog indicator (sustained positive net intake) --------------
    d["backlog_day"] = d["net_ma7"] > 0
    d["backlog_run"] = run_len(d["backlog_day"])
    d["backlog_sustained"] = d["backlog_run"] >= min_backlog_days

    # --- supplementary: census-based build-up (complements net intake) --
    d["census_change"] = d["hhs_care"].diff()
    d["census_ma7"] = d["census_change"].rolling(7, min_periods=3).mean()
    d["census_buildup"] = run_len(d["census_ma7"] > 0) >= min_backlog_days

    # --- strain & relief -----------------------------------------------
    thr = d["total_load"].quantile(strain_pct / 100)
    d["strain_threshold"] = thr
    d["high_load"] = d["load_ma14"] >= thr
    d["strain_run"] = run_len(d["high_load"])
    d["strain_window"] = d["strain_run"] >= min_strain_days
    d["load_trend7"] = d["load_ma7"].diff(7)
    d["relief_day"] = d["load_trend7"] < 0
    d["relief_window"] = run_len(d["relief_day"]) >= 7
    d["status"] = np.select(
        [d["strain_window"] & ~d["relief_window"], d["relief_window"]],
        ["Strain", "Relief"],
        default="Stable",
    )

    # --- robust anomaly flag on daily change (MAD z-score) ---------------
    g = d["growth_rate"]
    mad = (g - g.median()).abs().median()
    d["flag_growth_outlier"] = ((g - g.median()).abs() / (1.4826 * mad)) > 3.5 if mad > 0 else False
    return d


def list_windows(d, flag):
    """Table of contiguous periods where boolean column `flag` is True."""
    m = d[flag].fillna(False).astype(bool)
    grp = m.ne(m.shift(fill_value=False)).cumsum()
    rows = []
    for _, seg in d[m].groupby(grp[m]):
        rows.append({
            "start": seg.index[0].date(), "end": seg.index[-1].date(),
            "days": len(seg),
            "avg_total_load": round(seg["total_load"].mean()),
            "peak_total_load": round(seg["total_load"].max()),
            "avg_net_intake": round(seg["net_intake"].mean(), 1),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 5. TIME GRANULARITY
# ----------------------------------------------------------------------
FREQ = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}


def aggregate(d, freq="D"):
    """Resample to weekly/monthly. Stocks and flows are averaged (avg per
    day) so periods with different numbers of reported days stay comparable."""
    if freq == "D":
        return d
    cols = [c for c in ["cbp_custody", "hhs_care", "total_load", "apprehended",
                        "transferred", "discharged", "net_intake", "net_ma7",
                        "net_ma14", "load_ma7", "load_ma14"] if c in d.columns]
    out = d[cols].resample(freq).mean()
    out["cum_net_intake"] = d["cum_net_intake"].resample(freq).last()
    out["growth_rate"] = out["total_load"].pct_change() * 100
    out["strain_share"] = 100 * d["strain_window"].resample(freq).mean()
    out["backlog_share"] = 100 * d["backlog_sustained"].resample(freq).mean()
    out["census_change"] = d["census_change"].resample(freq).mean()
    return out


# ----------------------------------------------------------------------
# 6. KPIs
# ----------------------------------------------------------------------
def compute_kpis(d):
    nan = float("nan")
    if len(d) == 0:
        return {k: nan for k in ["total_latest", "total_avg", "peak_total",
                                 "net_pressure", "cum_net", "clvi",
                                 "backlog_rate", "backlog_share", "buildup_share", "dor"]}
    rep = d[d["reported"]]
    both = rep.dropna(subset=["transferred", "discharged"])
    sustained = d[d["backlog_sustained"]]
    return {
        "total_latest": d["total_load"].iloc[-1],
        "total_avg": d["total_load"].mean(),
        "peak_total": d["total_load"].max(),
        "peak_date": d["total_load"].idxmax(),
        "net_pressure": rep["net_intake"].mean(),
        "cum_net": d["net_intake_est"].sum(),
        "clvi": d["growth_rate"].std(),
        "backlog_rate": sustained["net_intake"].mean() if len(sustained) else 0.0,
        "backlog_share": 100 * d["backlog_sustained"].mean(),
        "buildup_share": 100 * d["census_buildup"].mean(),
        "dor": (both["discharged"].sum() / both["transferred"].sum())
        if both["transferred"].sum() > 0 else nan,
    }


def load_and_prepare(src, strain_pct=75):
    raw, info = load_raw(src)
    daily = build_daily(raw)
    daily = add_validation_flags(daily)
    daily = compute_metrics(daily, strain_pct=strain_pct)
    return daily, info

# ======================================================================
#                         STREAMLIT USER INTERFACE
# ======================================================================
import io
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="UAC Care Load Analytics", page_icon="🏥", layout="wide")

DEFAULT_CSV = Path(__file__).parent / "HHS_Unaccompanied_Alien_Children_Program.csv"
COLORS = {"cbp_custody": "#ed8936", "hhs_care": "#2b6cb0", "total_load": "#1a202c",
          "apprehended": "#e53e3e", "transferred": "#dd6b20", "discharged": "#38a169",
          "net_intake": "#6b46c1", "census_change": "#805ad5"}


def show(fig):
    """Render a Plotly figure full-width (works across Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def base_layout(fig, title, ytitle="Children", height=420):
    fig.update_layout(title=title, yaxis_title=ytitle, height=height, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def add_windows(fig, d, flag, color, name):
    """Shade contiguous windows where boolean column `flag` is True."""
    w = list_windows(d, flag)
    for _, r in w.iterrows():
        fig.add_vrect(x0=pd.Timestamp(r["start"]), x1=pd.Timestamp(r["end"]) + pd.Timedelta(days=1),
                      fillcolor=color, opacity=0.15, line_width=0, layer="below")
    return len(w)


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Processing data...")
def get_data(file_bytes, strain_pct):
    src = io.BytesIO(file_bytes) if file_bytes else DEFAULT_CSV
    return load_and_prepare(src, strain_pct=strain_pct)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.title("🏥 Controls")
upload = st.sidebar.file_uploader("Upload a newer HHS CSV (optional)", type="csv")
file_bytes = upload.getvalue() if upload else None

with st.sidebar.expander("Strain definition", expanded=False):
    strain_pct = st.slider("Strain threshold percentile", 50, 95, 75, 5,
                           help="A prolonged strain window = 14-day average total load stays at or above this "
                                "percentile of the whole period for at least 14 consecutive days.")

try:
    full, info = get_data(file_bytes, strain_pct)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the data: {e}")
    st.stop()

min_d, max_d = full.index.min().date(), full.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = pd.Timestamp(min_d), pd.Timestamp(max_d)

gran = st.sidebar.radio("Time granularity", list(FREQ), horizontal=True,
                        help="Weekly/Monthly views show the average per day within each period.")
load_metrics = st.sidebar.multiselect(
    "Load metrics", ["total_load", "cbp_custody", "hhs_care"], default=["total_load", "cbp_custody", "hhs_care"],
    format_func=lambda c: LABELS[c])
flow_metrics = st.sidebar.multiselect(
    "Flow metrics", ["apprehended", "transferred", "discharged", "net_intake"],
    default=["apprehended", "transferred", "discharged"], format_func=lambda c: LABELS[c])
rolling = st.sidebar.multiselect("Rolling averages (daily view)", [7, 14], default=[7],
                                 format_func=lambda w: f"{w}-day")
show_windows = st.sidebar.checkbox("Shade strain / backlog windows", value=True)

d = full.loc[start:end]
if len(d) < 2:
    st.warning("Please select a date range with at least two days.")
    st.stop()
agg = aggregate(d, FREQ[gran])
unit = "" if gran == "Daily" else " (avg per day)"

# ----------------------------------------------------------------------
# Header + KPI cards
# ----------------------------------------------------------------------
st.title("System Capacity & Care Load Analytics")
st.caption("HHS Unaccompanied Children (UAC) Program · CBP → HHS care pipeline · "
           f"{start:%d %b %Y} – {end:%d %b %Y} · {len(d):,} calendar days "
           f"({int(d['reported'].sum()):,} with a report)")

k = compute_kpis(d)
span = end - start
prev = full.loc[start - span - pd.Timedelta(days=1): start - pd.Timedelta(days=1)]
kp = compute_kpis(prev) if len(prev) > 5 else None


def delta(key, fmt="{:+,.0f}"):
    if kp is None or pd.isna(kp.get(key)) or pd.isna(k.get(key)):
        return None
    return fmt.format(k[key] - kp[key])


c = st.columns(5)
c[0].metric("Total children under care (latest)", f"{k['total_latest']:,.0f}", delta("total_latest"),
            delta_color="inverse", help="CBP custody + HHS care on the last day of the selected range. "
                                        "Delta vs the end of the previous equal-length period.")
c[1].metric("Average total load", f"{k['total_avg']:,.0f}", delta("total_avg"), delta_color="inverse")
c[2].metric("Net intake pressure", f"{k['net_pressure']:+.1f} /day", delta("net_pressure", "{:+.1f}"),
            delta_color="inverse", help="Average of (transfers into HHS − discharges). Positive = inflow exceeds outflow.")
c[3].metric("Volatility index", f"{k['clvi']:.2f}", delta("clvi", "{:+.2f}"), delta_color="inverse",
            help="Std. deviation of the daily % change in total load (percentage points). Higher = less stable.")
c[4].metric("Discharge offset ratio", f"{k['dor']:.2f}", delta("dor", "{:+.2f}"),
            help="Total discharged ÷ total transferred. Above 1 means discharges outpace new transfers.")
c = st.columns(4)
c[0].metric("Peak total load", f"{k['peak_total']:,.0f}", help=f"Reached on {k['peak_date']:%d %b %Y}")
c[1].metric("Backlog accumulation rate", f"{k['backlog_rate']:+.1f} /day", delta("backlog_rate", "{:+.1f}"),
            delta_color="inverse", help="Avg net intake on days inside a sustained backlog "
                                        "(7-day avg net intake > 0 for ≥ 7 days). 0 if none.")
c[2].metric("Days in sustained backlog", f"{k['backlog_share']:.0f}%", delta("backlog_share", "{:+.0f} pts"),
            delta_color="inverse")
c[3].metric("Days in prolonged strain", f"{100 * d['strain_window'].mean():.0f}%",
            help=f"14-day avg load ≥ {full['strain_threshold'].iloc[0]:,.0f} (P{strain_pct}) for ≥ 14 consecutive days")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
t1, t2, t3, t4, t5, t6 = st.tabs(["📊 System load overview", "⚖️ CBP vs HHS", "📈 Net intake & backlog",
                                  "🚨 Stress & volatility", "✅ Data quality", "🗂 Data explorer"])

# ---- 1. System load overview -----------------------------------------
with t1:
    fig = go.Figure()
    if "cbp_custody" in load_metrics and "hhs_care" in load_metrics:
        fig.add_trace(go.Scatter(x=agg.index, y=agg["hhs_care"], name=LABELS["hhs_care"], stackgroup="one",
                                 line=dict(width=0.5, color=COLORS["hhs_care"])))
        fig.add_trace(go.Scatter(x=agg.index, y=agg["cbp_custody"], name=LABELS["cbp_custody"], stackgroup="one",
                                 line=dict(width=0.5, color=COLORS["cbp_custody"])))
        if "total_load" in load_metrics:
            fig.add_trace(go.Scatter(x=agg.index, y=agg["total_load"], name=LABELS["total_load"],
                                     line=dict(color=COLORS["total_load"], width=1.8)))
    else:
        for m in load_metrics:
            fig.add_trace(go.Scatter(x=agg.index, y=agg[m], name=LABELS[m], line=dict(color=COLORS[m], width=2)))
    if gran == "Daily":
        for w in rolling:
            fig.add_trace(go.Scatter(x=d.index, y=d[f"load_ma{w}"], name=f"Total load, {w}-day avg",
                                     line=dict(dash="dash", width=1.5)))
    if show_windows:
        add_windows(fig, d, "strain_window", "#e53e3e", "Strain")
    base_layout(fig, f"Total system load{unit}")
    show(fig)
    st.caption("Red shading = prolonged strain window (14-day average load stays above the strain threshold).")

    if flow_metrics:
        fig = go.Figure()
        for m in flow_metrics:
            fig.add_trace(go.Scatter(x=agg.index, y=agg[m], name=LABELS[m], line=dict(color=COLORS[m], width=1.8)))
        if "net_intake" in flow_metrics:
            fig.add_hline(y=0, line_color="gray", line_width=1)
        base_layout(fig, f"Daily flows through the care pipeline{unit}", "Children per day", 380)
        show(fig)

# ---- 2. CBP vs HHS ---------------------------------------------------
with t2:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CBP custody (short-stay holding)",
                                                        "HHS care (long-stay shelters)"))
    fig.add_trace(go.Scatter(x=agg.index, y=agg["cbp_custody"], line=dict(color=COLORS["cbp_custody"]),
                             name="CBP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=agg.index, y=agg["hhs_care"], line=dict(color=COLORS["hhs_care"]),
                             name="HHS"), row=1, col=2)
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
    show(fig)

    share = 100 * agg["cbp_custody"] / agg["total_load"]
    fig = go.Figure(go.Scatter(x=agg.index, y=share, fill="tozeroy", line=dict(color=COLORS["cbp_custody"])))
    base_layout(fig, f"CBP share of total system load (%){unit}", "% of total", 300)
    show(fig)

    cc = st.columns(3)
    cc[0].metric("Avg children in CBP custody", f"{d['cbp_custody'].mean():,.0f}")
    cc[1].metric("Avg children in HHS care", f"{d['hhs_care'].mean():,.0f}")
    cc[2].metric("Avg CBP share of load", f"{100 * (d['cbp_custody'] / d['total_load']).mean():.1f}%")
    st.info("The two systems operate on very different scales: HHS holds thousands of children for weeks, "
            "while CBP custody is a short-stay buffer of tens to hundreds.")

# ---- 3. Net intake & backlog -------------------------------------------
with t3:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg.index, y=agg["net_intake"], name=LABELS["net_intake"], marker_color="#cbd5e0"))
    if gran == "Daily":
        for w, col in [(7, "#2b6cb0"), (14, "#c53030")]:
            if w in rolling:
                fig.add_trace(go.Scatter(x=d.index, y=d[f"net_ma{w}"], name=f"{w}-day avg",
                                         line=dict(color=col, width=2)))
    fig.add_hline(y=0, line_color="black", line_width=1)
    if show_windows:
        n = add_windows(fig, d, "backlog_sustained", "#e53e3e", "Backlog")
    base_layout(fig, f"Net daily intake (transfers − discharges){unit}", "Children per day")
    show(fig)
    st.caption("Positive = more children entering HHS care than leaving. Red shading = sustained backlog "
               "(7-day average net intake > 0 for at least 7 consecutive days).")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=agg.index, y=agg["cum_net_intake"], name="Cumulative net intake (est.)",
                             line=dict(color="#553c9a", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg.index, y=agg["hhs_care"], name="HHS census",
                             line=dict(color=COLORS["hhs_care"], width=2, dash="dot")), secondary_y=True)
    fig.update_layout(title=f"Cumulative net intake vs actual HHS census{unit}", height=380,
                      hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(title_text="Cumulative net intake", secondary_y=False)
    fig.update_yaxes(title_text="HHS census", secondary_y=True)
    show(fig)
    st.warning("**Read with care:** the HHS census does not reconcile tightly with (transfers − discharges) — HHS also "
               "receives children from sources other than CBP transfers. Treat net intake as a pipeline-pressure "
               "proxy and use the census trend and the census build-up indicator below as the check.")

    cc = st.columns(3)
    cc[0].metric("Sustained backlog days", f"{int(d['backlog_sustained'].sum()):,}")
    cc[1].metric("Census build-up days", f"{int(d['census_buildup'].sum()):,}",
                 help="Days where the 7-day average change in HHS census is positive for ≥ 7 consecutive days.")
    cc[2].metric("Cumulative net intake (est.)", f"{k['cum_net']:+,.0f}")
    bw = list_windows(d, "backlog_sustained")
    st.subheader("Sustained backlog periods")
    if len(bw):
        show_df(bw, hide_index=True)
    else:
        st.write("No sustained backlog in the selected range.")

# ---- 4. Stress & volatility --------------------------------------------
with t4:
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Scatter(x=agg.index, y=agg["growth_rate"], line=dict(color="#4299e1")))
        fig.add_hline(y=0, line_color="gray", line_width=1)
        base_layout(fig, f"Care load growth rate (% change){unit}", "% change", 340)
        show(fig)
    with b:
        fig = go.Figure(go.Scatter(x=d.index, y=d["growth_std14"], line=dict(color="#c05621")))
        base_layout(fig, "Volatility: 14-day std of daily growth (% pts)", "% points", 340)
        show(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["total_load"], name="Total load", line=dict(color="#a0aec0", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["load_ma14"], name="14-day avg", line=dict(color="black", width=2)))
    fig.add_hline(y=full["strain_threshold"].iloc[0], line_dash="dash", line_color="#c53030",
                  annotation_text=f"Strain threshold (P{strain_pct})", annotation_position="top left")
    add_windows(fig, d, "strain_window", "#e53e3e", "Strain")
    add_windows(fig, d, "relief_window", "#38a169", "Relief")
    base_layout(fig, "Strain (red) and relief (green) periods", height=420)
    show(fig)

    cc = st.columns(3)
    status_counts = d["status"].value_counts()
    for col, s in zip(cc, ["Strain", "Stable", "Relief"]):
        col.metric(f"{s} days", f"{int(status_counts.get(s, 0)):,}",
                   f"{100 * status_counts.get(s, 0) / len(d):.0f}% of range", delta_color="off")
    st.subheader("Prolonged strain windows")
    sw = list_windows(d, "strain_window")
    if len(sw):
        show_df(sw, hide_index=True)
    else:
        st.write("No prolonged strain window in the selected range.")
    st.caption("Relief = 7-day average load falling week-over-week for ≥ 7 consecutive days.")

# ---- 5. Data quality ---------------------------------------------------
with t5:
    st.subheader("Validation summary (full dataset)")
    show_df(validation_summary(full, info).astype(str).to_frame("Value"))
    r = reconciliation(full)
    st.subheader("Flow reconciliation check")
    st.write(f"Across **{r['pairs_compared']}** consecutive-day pairs, the correlation between the change in HHS census "
             f"and (transfers − discharges) is **{r['correlation']:.2f}**. Average daily census change is "
             f"**{r['mean_census_change']:+.1f}** versus **{r['mean_net_flow']:+.1f}** for net flow, so net intake "
             "is a proxy, not an exact census change.")
    cov = (100 * full["reported"].resample("MS").mean()).rename("coverage")
    fig = go.Figure(go.Bar(x=cov.index, y=cov, marker_color="#2b6cb0"))
    base_layout(fig, "Reporting coverage by month (% of days with a report)", "%", 300)
    fig.update_yaxes(range=[0, 105])
    show(fig)
    st.subheader("Days where transfers exceeded end-of-day CBP custody (flagged, not removed)")
    st.caption("This can be legitimate: custody is an end-of-day snapshot, while transfers occur during the day.")
    v = full.loc[full["flag_transfer_gt_cbp"], ["cbp_custody", "transferred", "hhs_care", "discharged"]]
    show_df(v)

# ---- 6. Data explorer --------------------------------------------------
with t6:
    cols = ["reported", "apprehended", "cbp_custody", "transferred", "hhs_care", "discharged", "total_load",
            "net_intake", "growth_rate", "net_ma7", "load_ma7", "backlog_sustained", "strain_window", "status"]
    out = agg.copy() if gran != "Daily" else d[cols]
    show_df(out.round(2))
    st.download_button("⬇️ Download this view as CSV", out.round(3).to_csv().encode(),
                       file_name=f"uac_{gran.lower()}_{start:%Y%m%d}_{end:%Y%m%d}.csv", mime="text/csv")

st.divider()
st.caption("Metric definitions: Total load = CBP custody + HHS care · Net intake = transfers − discharges · "
           "Growth rate = day-over-day % change in total load · Backlog = 7-day avg net intake > 0 for ≥ 7 days · "
           "Strain = 14-day avg load ≥ threshold for ≥ 14 days.")
