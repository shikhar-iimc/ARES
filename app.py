import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import pickle
import time

st.set_page_config(
    page_title="ARES — Conflict Intelligence",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.html("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background: #FFFFFF !important; }
.block-container { padding: 0 40px 80px 40px !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"], section[data-testid="stSidebar"] { display: none !important; }
.top-bar { position:fixed; top:0; left:0; right:0; height:3px; background:#C41E3A; z-index:9999; }
html, body, p, div, span, h1, h2, h3, label, input, button,
[class*="css"], .stMarkdown, .stText, .element-container {
  font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
  color: #0A0A0A !important;
}
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important; background: transparent !important;
  border-bottom: 1px solid #E5E5E5 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: 0.1em !important; color: #333333 !important;
  background: transparent !important; border: none !important;
  border-radius: 0 !important; padding: 12px 22px !important;
  text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
  color: #0A0A0A !important;
  border-bottom: 2px solid #C41E3A !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stCheckbox label, .stCheckbox label p,
.stCheckbox span[data-testid="stMarkdownContainer"] p {
  color: #0A0A0A !important; font-size: 13px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}
.stRadio label, .stRadio label p,
.stRadio div[data-testid="stMarkdownContainer"] p {
  color: #0A0A0A !important; font-size: 13px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}
.stSlider label, .stSlider label p { color: #0A0A0A !important; }
.stTextInput label, .stTextInput label p { color: #0A0A0A !important; }
.stTextInput input {
  border: 1px solid #E5E5E5 !important; border-radius: 0 !important;
  background: #FFFFFF !important; color: #0A0A0A !important;
  font-size: 13px !important;
}
[data-testid="metric-container"] {
  background: #FFFFFF !important; border: 1px solid #E5E5E5 !important;
  border-radius: 0 !important; padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 10px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.12em !important;
  color: #6B6B6B !important;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 26px !important; font-weight: 600 !important;
  color: #0A0A0A !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; color: #6B6B6B !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }
.stButton button {
  background: #FFFFFF !important; color: #0A0A0A !important;
  border: 1px solid #CCCCCC !important; border-radius: 0 !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: 0.06em !important; padding: 8px 20px !important;
  box-shadow: none !important;
}
.stButton button:hover {
  border-color: #C41E3A !important; color: #C41E3A !important;
  background: #FFFFFF !important; box-shadow: none !important;
}
.stButton button:focus {
  border-color: #C41E3A !important; color: #C41E3A !important;
  box-shadow: none !important; outline: none !important;
}
</style>
<div class="top-bar"></div>
""")

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("ares_data.pkl", "rb") as f:
        return pickle.load(f)

data           = load_data()
SIM_LOG        = data["sim_log"]
G_INITIAL      = data["g_initial"]
G_FINAL        = data["g_final"]
EVENTS_DF      = data["events_df"]
EDGE_SNAPSHOTS = data.get("edge_snapshots", None)
N_TICKS        = len(SIM_LOG)

# ── COMPUTE PERCENTAGE MIGHT (fix for national power chart) ───
def compute_pct_might(sim_log, country):
    col = f"{country.lower()}_might"
    if col not in sim_log.columns:
        return None
    start = sim_log.iloc[0][col]
    if start == 0:
        return None
    return (sim_log[col] / start * 100).round(2)

iran_pct   = compute_pct_might(SIM_LOG, "Iran")
israel_pct = compute_pct_might(SIM_LOG, "Israel")
usa_pct    = compute_pct_might(SIM_LOG, "USA")

# ── PLOT BASE ─────────────────────────────────────────────────
PB = dict(
    font=dict(family="IBM Plex Sans", size=12, color="#0A0A0A"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    margin=dict(l=0, r=0, t=28, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font=dict(size=11, color="#0A0A0A"))
)
AX = dict(gridcolor="#F0F0F0", linecolor="#E5E5E5",
          tickfont=dict(family="IBM Plex Mono", size=10, color="#555555"))

def sec(label):
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:#6B6B6B;margin:28px 0 12px 0;'
        f'padding-bottom:8px;border-bottom:1px solid #EEEEEE">{label}</div>',
        unsafe_allow_html=True)

def mono(txt, color="#6B6B6B", size="12px"):
    return (f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:{size};color:{color}">{txt}</span>')

def progress_bar(pct):
    st.markdown(
        f'<div style="background:#F0F0F0;height:3px;width:100%;margin-bottom:20px">'
        f'<div style="background:#C41E3A;height:3px;width:{pct}%"></div></div>',
        unsafe_allow_html=True)

def add_phase_lines(fig):
    for d, l in [("2024-04-13","First direct exchange"),
                 ("2025-06-13","Twelve-Day War"),
                 ("2026-02-28","Operation Epic Fury"),
                 ("2026-04-08","Pakistan ceasefire")]:
        fig.add_vline(x=d, line_dash="dot", line_color="#CCCCCC",
                      line_width=1, annotation_text=l,
                      annotation_textangle=-90,
                      annotation_font=dict(size=9, color="#999999"))

def play_controls(key_prefix, n_ticks):
    tick_key    = f"{key_prefix}_tick"
    playing_key = f"{key_prefix}_playing"
    if tick_key    not in st.session_state: st.session_state[tick_key]    = n_ticks - 1
    if playing_key not in st.session_state: st.session_state[playing_key] = False

    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        lbl = "⏸  Pause" if st.session_state[playing_key] else "▶  Play"
        if st.button(lbl, key=f"{key_prefix}_play_btn"):
            st.session_state[playing_key] = not st.session_state[playing_key]
            if st.session_state[playing_key]:
                st.session_state[tick_key] = 0
    with b2:
        if st.button("↺  Reset", key=f"{key_prefix}_reset_btn"):
            st.session_state[tick_key]    = 0
            st.session_state[playing_key] = False

    tick = st.slider(
        "Event index", min_value=0, max_value=n_ticks - 1,
        value=st.session_state[tick_key],
        disabled=st.session_state[playing_key],
        key=f"{key_prefix}_slider"
    )
    if not st.session_state[playing_key]:
        st.session_state[tick_key] = tick
    return st.session_state[tick_key], playing_key

def build_network_figure(G_ref, edge_data, thresh, title):
    ns_sorted = sorted(G_ref.nodes(data=True),
                       key=lambda x: x[1].get("alignment", 0), reverse=True)
    nn  = len(ns_sorted)
    pos = {}
    for i, (nd, _) in enumerate(ns_sorted):
        a = 2 * np.pi * i / nn
        pos[nd] = (np.cos(a), np.sin(a))

    etrs = []
    for (a, b), r in edge_data.items():
        if abs(r) < thresh: continue
        if a not in pos or b not in pos: continue
        x0,y0=pos[a]; x1,y1=pos[b]
        op  = min(0.9, 0.15 + abs(r)*0.75)
        col = (f"rgba(27,122,62,{op})" if r >= 0
               else f"rgba(196,30,58,{op})")
        etrs.append(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None], mode="lines",
            line=dict(width=max(0.3, abs(r)*2.5), color=col),
            hoverinfo="none", showlegend=False))

    nx_l,ny_l,nt_l,nh_l,nc_l,nz_l=[],[],[],[],[],[]
    for nd, nd_d in G_ref.nodes(data=True):
        x,y=pos[nd]
        nx_l.append(x); ny_l.append(y); nt_l.append(nd)
        al=nd_d.get("alignment",0); m=nd_d.get("might",0.1)
        camp="US" if al>0.15 else ("Iran" if al<-0.15 else "Neutral")
        nh_l.append(f"<b>{nd}</b><br>Camp:{camp}<br>Might:{m:.3f}<br>Alignment:{al:+.3f}")
        nc_l.append("#1B3A6B" if al>0.15 else ("#C41E3A" if al<-0.15 else "#888888"))
        nz_l.append(12 + 35*float(m))

    ntr = go.Scatter(
        x=nx_l, y=ny_l, mode="markers+text",
        text=nt_l, textposition="top center",
        textfont=dict(family="IBM Plex Sans", size=10, color="#0A0A0A"),
        hovertext=nh_l, hoverinfo="text",
        marker=dict(color=nc_l, size=nz_l,
                    line=dict(width=1.5, color="#FFFFFF")),
        showlegend=False)

    fig = go.Figure(data=etrs + [ntr])
    fig.update_layout(height=600, paper_bgcolor="#FFFFFF",
                      plot_bgcolor="#FFFFFF",
                      margin=dict(l=0,r=0,t=36,b=0),
                      font=dict(family="IBM Plex Sans"),
                      title=dict(text=title,
                                 font=dict(size=13,color="#6B6B6B",
                                           family="IBM Plex Sans"),x=0))
    fig.update_xaxes(showgrid=False,zeroline=False,
                     showticklabels=False,scaleanchor="y")
    fig.update_yaxes(showgrid=False,zeroline=False,showticklabels=False)
    return fig

# ── HEADER ────────────────────────────────────────────────────
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown(
        '<div style="padding:44px 0 20px 0;border-bottom:1px solid #E5E5E5">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:12px;'
        'font-weight:500;letter-spacing:0.3em;color:#C41E3A;margin-bottom:6px">'
        '▲ ARES</div>'
        '<div style="font-size:26px;font-weight:700;color:#0A0A0A;'
        'letter-spacing:-0.5px;margin-bottom:6px">'
        'Adaptive Relationship &amp; Event Simulator</div>'
        '<div style="font-size:13px;color:#6B6B6B">'
        'US–Iran–Israel Conflict Network &nbsp;·&nbsp; Oct 2023 → Aug 2026'
        ' &nbsp;·&nbsp; Structural Balance Theory (Marvel–Kleinberg–Strogatz 2011)'
        '</div></div>',
        unsafe_allow_html=True)
with h2:
    st.markdown(
        '<div style="padding:44px 0 20px 0;border-bottom:1px solid #E5E5E5;'
        'text-align:right">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        'color:#9B9B9B;line-height:2.2;letter-spacing:0.06em">'
        'IIM CALCUTTA<br>SNA · GROUP 5<br>MBA BATCH 62</div></div>',
        unsafe_allow_html=True)

T = st.tabs(["OVERVIEW","NETWORK","TRAJECTORIES","MEDIATOR ANALYSIS","EVENT LOG"])

# ═══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════
with T[0]:
    n_ev = len(EVENTS_DF)
    n_h  = int((EVENTS_DF["goldstein"] < 0).sum())
    n_d  = int((EVENTS_DF["goldstein"] > 0).sum())
    ir_s = float(SIM_LOG.iloc[0]["iran_might"])
    ir_e = float(SIM_LOG.iloc[-1]["iran_might"])
    ir_l = (1 - ir_e/ir_s)*100 if ir_s > 0 else 0
    fmed = SIM_LOG.iloc[-1]["top_mediator"]

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Events",        str(n_ev),       "Oct 2023 – Aug 2026")
    m2.metric("Hostile Events",      str(n_h),        f"{n_h/n_ev*100:.0f}% of total")
    m3.metric("Diplomatic Events",   str(n_d),        f"{n_d/n_ev*100:.0f}% of total")
    m4.metric("Iran Power Loss",     f"{ir_l:.0f}%",  "vs initial capacity")
    m5.metric("Structural Mediator", fmed,            "by network position")

    sec("Conflict Phases")
    phases = [
        ("PHASE I",   "Oct 2023 – Dec 2023",
         "Hamas Oct 7 attack triggers regional cascade. Houthis and Hezbollah activate. US expands Iran sanctions."),
        ("PHASE II",  "Jan 2024 – Oct 2024",
         "First direct Iran–Israel exchanges. 300+ drones and missiles. Nasrallah and Sinwar killed."),
        ("PHASE III", "Nov 2024 – Feb 2026",
         "Lebanon ceasefire. Assad falls. Twelve-Day War (Jun 2025): US joins Israel strikes on Iran nuclear sites."),
        ("PHASE IV",  "Feb 2026 – Present",
         "Operation Epic Fury: 900 strikes, Khamenei killed. Strait of Hormuz crisis. Pakistan brokers ceasefire."),
    ]
    for pid, dates, desc in phases:
        c1,c2,c3 = st.columns([1,2,6])
        c1.markdown(mono(pid,"#C41E3A","11px"), unsafe_allow_html=True)
        c2.markdown(mono(dates,"#6B6B6B","12px"), unsafe_allow_html=True)
        c3.markdown(f'<span style="font-size:13px;color:#0A0A0A">{desc}</span>',
                    unsafe_allow_html=True)
        st.markdown('<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',
                    unsafe_allow_html=True)

    # ── FIXED: National Power as % of initial ──
    sec("National Power Trajectory  (% of initial capacity)")
    fig_m = go.Figure()

    for series, label, col, dash in [
        (iran_pct,   "Iran",   "#C41E3A", "solid"),
        (israel_pct, "Israel", "#1B3A6B", "solid"),
        (usa_pct,    "USA",    "#1B7A3E", "dash"),
    ]:
        if series is not None:
            fig_m.add_trace(go.Scatter(
                x=SIM_LOG["date"], y=series, name=label,
                mode="lines", line=dict(color=col, width=2, dash=dash),
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.1f}}% of initial<extra></extra>"
            ))

    # add 100% reference line
    fig_m.add_hline(y=100, line_dash="dot", line_color="#DDDDDD",
                    line_width=1, annotation_text="Starting capacity (100%)",
                    annotation_position="bottom right",
                    annotation_font=dict(size=9, color="#AAAAAA"))

    add_phase_lines(fig_m)
    fig_m.update_layout(**PB, height=300)
    fig_m.update_xaxes(**AX, title_text="")
    fig_m.update_yaxes(**AX, title_text="% of initial capacity",
                       range=[0, 110])
    st.plotly_chart(fig_m, use_container_width=True)

    # caption explaining the chart
    st.markdown(
        '<div style="font-size:11px;color:#9B9B9B;margin-top:-16px;margin-bottom:20px">'
        'All three countries start at 100%. Iran degrades most severely from repeated '
        'direct strikes across all four phases. Israel bears offensive costs from '
        'sustained operations. USA degrades from prolonged campaign resource expenditure.'
        '</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 2 — NETWORK (animated)
# ═══════════════════════════════════════════════
with T[1]:
    cc, cm = st.columns([1, 4])
    with cc:
        sec("Controls")
        thresh = st.slider("Edge threshold", 0.0, 1.0, 0.25, 0.05, key="net_thresh")
        st.markdown("<br>", unsafe_allow_html=True)
        for dot_col, lbl in [("#1B3A6B","US-aligned"),
                              ("#C41E3A","Iran-aligned"),
                              ("#888888","Straddler")]:
            st.markdown(
                f'<div style="font-size:12px;color:#0A0A0A;margin-bottom:6px">'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{dot_col};margin-right:8px;'
                f'vertical-align:middle"></span>{lbl}</div>',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for line_col, lbl in [("#1B7A3E","Friendly"),("#C41E3A","Hostile")]:
            st.markdown(
                f'<div style="font-size:12px;color:#0A0A0A;margin-bottom:6px">'
                f'<span style="color:{line_col};font-weight:700;margin-right:8px">'
                f'—</span>{lbl}</div>',
                unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#6B6B6B;margin-top:8px">'
                    'Node size = Might</div>', unsafe_allow_html=True)

    with cm:
        if EDGE_SNAPSHOTS is not None:
            net_tick, net_playing_key = play_controls("net", N_TICKS)
            pct_bar = int((net_tick / max(N_TICKS-1, 1)) * 100)
            progress_bar(pct_bar)

            snap_row = SIM_LOG.iloc[net_tick]
            snap_dt  = snap_row["date"]
            snap_ev  = snap_row["event"]

            st.markdown(
                f'<div style="font-size:12px;color:#6B6B6B;margin-bottom:12px">'
                f'{mono(snap_dt,"#C41E3A","12px")} &nbsp;·&nbsp; {snap_ev}</div>',
                unsafe_allow_html=True)

            edge_data = EDGE_SNAPSHOTS[net_tick]
            fig_n = build_network_figure(
                G_INITIAL, edge_data, thresh,
                f"Conflict Network · {snap_dt}"
            )
            st.plotly_chart(fig_n, use_container_width=True)

            # relationship changes table
            if net_tick > 0:
                prev_data = EDGE_SNAPSHOTS[net_tick - 1]
                changes = []
                for k, v in edge_data.items():
                    prev_v = prev_data.get(k, v)
                    delta  = v - prev_v
                    if abs(delta) > 0.05:
                        a, b = k
                        changes.append((a, b, prev_v, v, delta))
                changes.sort(key=lambda x: abs(x[4]), reverse=True)
                if changes:
                    sec("Largest Relationship Changes This Event")
                    hc = st.columns([2,2,2,2,2])
                    for h, hl in zip(hc,["Country A","Country B","Before","After","Change"]):
                        h.markdown(
                            f'<span style="font-size:10px;font-weight:700;'
                            f'text-transform:uppercase;letter-spacing:0.1em;'
                            f'color:#6B6B6B">{hl}</span>',
                            unsafe_allow_html=True)
                    st.markdown('<hr style="margin:4px 0 0 0;border:none;'
                                'border-top:1px solid #E5E5E5">',
                                unsafe_allow_html=True)
                    for a,b,pv,cv,dv in changes[:6]:
                        col = "#C41E3A" if dv < 0 else "#1B7A3E"
                        sign = "+" if dv >= 0 else ""
                        rc = st.columns([2,2,2,2,2])
                        rc[0].markdown(f'<span style="font-size:12px;font-weight:600;color:#0A0A0A">{a}</span>',unsafe_allow_html=True)
                        rc[1].markdown(f'<span style="font-size:12px;font-weight:600;color:#0A0A0A">{b}</span>',unsafe_allow_html=True)
                        rc[2].markdown(mono(f"{pv:+.3f}","#888888","12px"),unsafe_allow_html=True)
                        rc[3].markdown(mono(f"{cv:+.3f}","#0A0A0A","12px"),unsafe_allow_html=True)
                        rc[4].markdown(
                            f'<span style="font-family:IBM Plex Mono,monospace;'
                            f'font-size:12px;font-weight:700;color:{col}">'
                            f'{sign}{dv:.3f}</span>',
                            unsafe_allow_html=True)
                        st.markdown('<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',
                                    unsafe_allow_html=True)

            if st.session_state[net_playing_key]:
                time.sleep(0.5)
                if st.session_state["net_tick"] < N_TICKS - 1:
                    st.session_state["net_tick"] += 1
                else:
                    st.session_state[net_playing_key] = False
                st.rerun()
        else:
            net_ch = st.radio("Network state",
                              ["Initial — Oct 2023","Final — Aug 2026"],
                              label_visibility="collapsed", key="net_static")
            G_sh = G_INITIAL if "Initial" in net_ch else G_FINAL
            edge_data = {(a,b): G_sh[a][b]["relationship"]
                         for a,b in G_sh.edges()}
            lbl = ("Initial Network — Oct 2023" if "Initial" in net_ch
                   else "Final Network — Aug 2026")
            fig_n = build_network_figure(G_sh, edge_data, thresh, lbl)
            st.plotly_chart(fig_n, use_container_width=True)
            st.info("Upload updated ares_data.pkl with edge_snapshots for animated mode.")

# ═══════════════════════════════════════════════
# TAB 3 — TRAJECTORIES (fixed)
# ═══════════════════════════════════════════════
with T[2]:

    # ── FIXED: Balance Energy — show both measures prominently ──
    sec("Network Tension Over Time")

    # check if energy has meaningful variation
    be_range = SIM_LOG["balance_energy"].max() - SIM_LOG["balance_energy"].min()
    fu_range = SIM_LOG["frac_unbalanced"].max() - SIM_LOG["frac_unbalanced"].min()

    # plot frac_unbalanced on left axis (primary — this has variation)
    # plot balance_energy on right axis (secondary)
    fig_t = go.Figure()

    # frac unbalanced — primary story (red, solid)
    fig_t.add_trace(go.Scatter(
        x=SIM_LOG["date"], y=SIM_LOG["frac_unbalanced"] * 100,
        name="Frustrated Triads (%)",
        line=dict(color="#C41E3A", width=2.5),
        yaxis="y1",
        hovertemplate="<b>Frustrated Triads</b><br>%{x}<br>%{y:.1f}% of triads<extra></extra>"
    ))

    # balance energy — secondary (navy, dashed)
    fig_t.add_trace(go.Scatter(
        x=SIM_LOG["date"], y=SIM_LOG["balance_energy"],
        name="Balance Energy",
        line=dict(color="#1B3A6B", width=1.5, dash="dash"),
        yaxis="y2",
        hovertemplate="<b>Balance Energy</b><br>%{x}<br>%{y:.4f}<extra></extra>"
    ))

    # top mediator score (gray dotted)
    fig_t.add_trace(go.Scatter(
        x=SIM_LOG["date"], y=SIM_LOG["top_score"],
        name="Top Mediator Score",
        line=dict(color="#AAAAAA", width=1.5, dash="dot"),
        yaxis="y1",
        hovertemplate="<b>Top Mediator Score</b><br>%{x}<br>%{y:.4f}<extra></extra>"
    ))

    add_phase_lines(fig_t)

    fig_t.update_layout(
        **PB, height=360,
        yaxis=dict(
            title="Frustrated Triads (%) / Mediator Score",
            gridcolor="#F0F0F0", linecolor="#E5E5E5",
            tickfont=dict(family="IBM Plex Mono", size=10, color="#555555"),
            side="left"
        ),
        yaxis2=dict(
            title="Balance Energy",
            gridcolor="#F0F0F0", linecolor="#E5E5E5",
            tickfont=dict(family="IBM Plex Mono", size=10, color="#1B3A6B"),
            side="right", overlaying="y",
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11, color="#0A0A0A"))
    )
    fig_t.update_xaxes(**AX, title_text="")
    st.plotly_chart(fig_t, use_container_width=True)

    # interpretation caption
    st.markdown(
        '<div style="font-size:11px;color:#9B9B9B;margin-top:-16px;margin-bottom:24px">'
        '<b style="color:#C41E3A">Red line (Frustrated Triads):</b> '
        'percentage of all 1,330 triangles in contradiction — higher = more network tension. '
        'Spikes at escalation events; drops when camps crystallise. &nbsp;'
        '<b style="color:#1B3A6B">Navy dashed (Balance Energy):</b> '
        'more negative = more stable. Near-zero indicates relationship magnitudes '
        'have decayed — a known model limitation from aggressive decay parameters.'
        '</div>',
        unsafe_allow_html=True)

    # ── NATIONAL POWER % in trajectories too ──
    sec("National Power Degradation  (% of initial capacity)")
    fig_pow = go.Figure()
    for series, label, col, dash in [
        (iran_pct,   "Iran",   "#C41E3A", "solid"),
        (israel_pct, "Israel", "#1B3A6B", "solid"),
        (usa_pct,    "USA",    "#1B7A3E", "dash"),
    ]:
        if series is not None:
            fig_pow.add_trace(go.Scatter(
                x=SIM_LOG["date"], y=series, name=label,
                mode="lines", line=dict(color=col, width=2, dash=dash),
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>"
            ))
    fig_pow.add_hline(y=100, line_dash="dot", line_color="#DDDDDD", line_width=1)
    add_phase_lines(fig_pow)
    fig_pow.update_layout(**PB, height=260)
    fig_pow.update_xaxes(**AX, title_text="")
    fig_pow.update_yaxes(**AX, title_text="% of initial capacity", range=[0,110])
    st.plotly_chart(fig_pow, use_container_width=True)

    sec("Mediator Leadership — Events as Top Mediator")
    mc2 = SIM_LOG["top_mediator"].value_counts().reset_index()
    mc2.columns = ["country","count"]
    mc2 = mc2.sort_values("count", ascending=True)
    top_c = mc2.iloc[-1]["country"]
    fig_b = go.Figure(go.Bar(
        x=mc2["count"], y=mc2["country"], orientation="h",
        marker=dict(color=["#C41E3A" if c==top_c else "#E5E5E5"
                           for c in mc2["country"]], line=dict(width=0)),
        text=mc2["count"], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#0A0A0A")
    ))
    fig_b.update_layout(**PB, height=320, showlegend=False)
    fig_b.update_xaxes(**AX, title_text="Events as top mediator")
    fig_b.update_yaxes(**AX, title_text="")
    st.plotly_chart(fig_b, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4 — MEDIATOR ANALYSIS (animated)
# ═══════════════════════════════════════════════
with T[3]:
    med_tick, med_playing_key = play_controls("med", N_TICKS)
    pct_bar = int((med_tick / max(N_TICKS-1, 1)) * 100)
    progress_bar(pct_bar)

    sel     = SIM_LOG.iloc[med_tick]
    top3    = sel["top3"]
    snap_dt = sel["date"]
    snap_ev = sel["event"]

    sec(f"Mediator Ranking · {snap_dt}")
    st.markdown(
        f'<div style="font-size:12px;color:#6B6B6B;margin-bottom:20px">'
        f'{snap_ev}</div>',
        unsafe_allow_html=True)

    # podium circles
    if len(top3) >= 3:
        p1,p2,p3 = st.columns(3)
        for col_w,(name,score),border,sz,rnk in [
            (p1,top3[0],"#1B3A6B","120px","RANK 01"),
            (p2,top3[1],"#888888","100px","RANK 02"),
            (p3,top3[2],"#C8A87A","100px","RANK 03"),
        ]:
            with col_w:
                st.markdown(
                    f'<div style="display:flex;flex-direction:column;'
                    f'align-items:center;padding:16px 0">'
                    f'<div style="width:{sz};height:{sz};border-radius:50%;'
                    f'border:2.5px solid {border};background:#FFFFFF;'
                    f'display:flex;flex-direction:column;align-items:center;'
                    f'justify-content:center;margin-bottom:10px">'
                    f'<div style="font-size:15px;font-weight:700;color:#0A0A0A;'
                    f'text-align:center;line-height:1.2">{name}</div>'
                    f'<div style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:10px;color:#6B6B6B;margin-top:4px">'
                    f'{score:.4f}</div></div>'
                    f'<div style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:10px;color:#9B9B9B;letter-spacing:0.1em">'
                    f'{rnk}</div></div>',
                    unsafe_allow_html=True)

    # score bars
    sec("All Mediator Scores at This Moment")
    max_score = max((s for _,s in top3), default=1)
    for name,score in top3:
        bw  = int((score/max_score)*300) if max_score > 0 else 0
        col = "#C41E3A" if name==top3[0][0] else "#1B3A6B"
        st.markdown(
            f'<div style="margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:12px;margin-bottom:3px">'
            f'<span style="color:#0A0A0A;font-weight:'
            f'{"700" if name==top3[0][0] else "400"}">{name}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;color:#6B6B6B">'
            f'{score:.4f}</span></div>'
            f'<div style="background:#F0F0F0;height:3px;width:100%">'
            f'<div style="background:{col};height:3px;width:{bw}px"></div>'
            f'</div></div>',
            unsafe_allow_html=True)

    # key moments
    sec("Mediator at Key Moments")
    key_moments = [
        ("2023-10-07","Oct 7 — Hamas attack"),
        ("2024-04-13","First direct Iran–Israel exchange"),
        ("2024-10-01","Iran fires 180 ballistic missiles"),
        ("2025-06-13","Twelve-Day War begins"),
        ("2026-02-28","Operation Epic Fury"),
        ("2026-04-08","Pakistan brokers ceasefire"),
    ]
    hcols = st.columns([2,2,1,4])
    for hc,hl in zip(hcols,["Date","Mediator","Score","Event"]):
        hc.markdown(
            f'<span style="font-size:10px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#6B6B6B">{hl}</span>',
            unsafe_allow_html=True)
    st.markdown('<hr style="margin:4px 0 0 0;border:none;border-top:1px solid #E5E5E5">',
                unsafe_allow_html=True)
    for d,label in key_moments:
        match = SIM_LOG[SIM_LOG["date"] >= d]
        if not match.empty:
            row = match.iloc[0]
            rc  = st.columns([2,2,1,4])
            rc[0].markdown(mono(d,"#6B6B6B","12px"),unsafe_allow_html=True)
            rc[1].markdown(
                f'<span style="font-weight:600;font-size:13px;color:#1B3A6B">'
                f'{row["top_mediator"]}</span>',unsafe_allow_html=True)
            rc[2].markdown(mono(f'{row["top_score"]:.4f}',"#888888","12px"),
                           unsafe_allow_html=True)
            rc[3].markdown(
                f'<span style="font-size:12px;color:#0A0A0A">{label}</span>',
                unsafe_allow_html=True)
        st.markdown('<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',
                    unsafe_allow_html=True)

    # finding
    st.markdown(
        '<div style="background:#F8F8F8;border-left:3px solid #1B3A6B;'
        'padding:16px 20px;margin:24px 0;font-size:13px;line-height:1.7;'
        'color:#0A0A0A">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.15em;color:#1B3A6B;margin-bottom:8px">Key Finding</div>'
        'Our structural model identifies <b>India and China</b> as best-positioned '
        'mediators by network topology — large states maintaining non-hostile ties '
        'to both camps throughout the conflict. Real-world mediation was carried '
        'out by <b>Qatar</b> (2023 hostage deal, 2025 Gaza ceasefire), '
        '<b>Oman</b> (2025 US–Iran back-channel, Strait of Hormuz MOU), and '
        '<b>Pakistan</b> (Apr 2026 ceasefire, Jun 2026 14-point MOU). '
        'This divergence shows structural position is '
        '<em>necessary but not sufficient</em> for mediation — political trust, '
        'geographic leverage, and pre-existing back-channels also matter.</div>',
        unsafe_allow_html=True)

    col_mod,col_real = st.columns(2)
    with col_mod:
        sec("Model Output")
        for rank,(name,score) in enumerate(SIM_LOG.iloc[-1]["top3"],1):
            bw = min(int(score*900),300)
            st.markdown(
                f'<div style="margin-bottom:14px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:12px;margin-bottom:4px">'
                f'<b style="color:#0A0A0A">#{rank} {name}</b>'
                f'<span style="font-family:IBM Plex Mono,monospace;color:#6B6B6B">'
                f'{score:.4f}</span></div>'
                f'<div style="background:#F0F0F0;height:3px;width:100%">'
                f'<div style="background:#1B3A6B;height:3px;width:{bw}px">'
                f'</div></div></div>',
                unsafe_allow_html=True)
    with col_real:
        sec("Real-World Mediators")
        for name,roles in [
            ("Qatar",   "Nov 2023 hostage deal · Jan 2025 Gaza ceasefire"),
            ("Oman",    "2025 US–Iran back-channel · Strait of Hormuz MOU"),
            ("Pakistan","Apr 2026 ceasefire · Jun 2026 14-point MOU"),
        ]:
            st.markdown(
                f'<div style="margin-bottom:16px">'
                f'<div style="font-size:14px;font-weight:600;color:#0A0A0A;'
                f'margin-bottom:3px">{name}</div>'
                f'<div style="font-size:12px;color:#6B6B6B;line-height:1.6">'
                f'{roles}</div></div>',
                unsafe_allow_html=True)

    # auto-advance
    if st.session_state[med_playing_key]:
        time.sleep(0.4)
        if st.session_state["med_tick"] < N_TICKS - 1:
            st.session_state["med_tick"] += 1
        else:
            st.session_state[med_playing_key] = False
        st.rerun()

# ═══════════════════════════════════════════════
# TAB 5 — EVENT LOG
# ═══════════════════════════════════════════════
with T[4]:
    f1,f2,f3 = st.columns(3)
    with f1: show_h = st.checkbox("Hostile events", True)
    with f2: show_d = st.checkbox("Diplomatic events", True)
    with f3: srch   = st.text_input("Filter by actor", placeholder="e.g. Iran")

    filt = EVENTS_DF.copy()
    if not show_h: filt = filt[filt["goldstein"] >= 0]
    if not show_d: filt = filt[filt["goldstein"] <= 0]
    if srch:
        mask = (filt["actor1"].str.contains(srch,case=False,na=False) |
                filt["actor2"].str.contains(srch,case=False,na=False))
        filt = filt[mask]

    sec(f"{len(filt)} Events")

    # goldstein chart — clip to [-10, +10] to avoid stacking artifact
    fig_g = go.Figure()
    fig_g.add_bar(
        x=filt["date"].astype(str),
        y=filt["goldstein"].clip(-10, 10),
        marker_color=["#C41E3A" if g<0 else "#1B7A3E"
                      for g in filt["goldstein"]],
        hovertext=(filt["actor1"]+" → "+filt["actor2"]
                   +"<br>"+filt["description"]
                   +"<br>Goldstein: "+filt["goldstein"].astype(str)),
        hoverinfo="text"
    )
    fig_g.update_layout(**PB, height=220, showlegend=False, bargap=0.15)
    fig_g.update_xaxes(**AX, title_text="")
    fig_g.update_yaxes(**AX, title_text="Goldstein Scale",
                       range=[-11, 11])
    st.plotly_chart(fig_g, use_container_width=True)

    # note about multiple same-date events
    st.markdown(
        '<div style="font-size:11px;color:#9B9B9B;margin-top:-16px;margin-bottom:16px">'
        'Goldstein scale clipped to [−10, +10]. Multiple events on the same date '
        '(e.g. Feb 28 2026) shown as separate bars, not stacked.'
        '</div>',
        unsafe_allow_html=True)

    # table header
    hcols2 = st.columns([2,1.5,1.5,4,1])
    for hc,hl in zip(hcols2,["Date","Actor 1","Actor 2","Description","Score"]):
        hc.markdown(
            f'<span style="font-size:10px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#6B6B6B">{hl}</span>',
            unsafe_allow_html=True)
    st.markdown('<hr style="margin:4px 0 0 0;border:none;border-top:2px solid #E5E5E5">',
                unsafe_allow_html=True)

    for _,row in filt.iterrows():
        g=row["goldstein"]; sign="+" if g>=0 else ""; gcol="#C41E3A" if g<0 else "#1B7A3E"
        rc=st.columns([2,1.5,1.5,4,1])
        rc[0].markdown(mono(str(row["date"].date()),"#6B6B6B","12px"),unsafe_allow_html=True)
        rc[1].markdown(f'<span style="font-weight:500;font-size:12px;color:#0A0A0A">{row["actor1"]}</span>',unsafe_allow_html=True)
        rc[2].markdown(f'<span style="font-weight:500;font-size:12px;color:#0A0A0A">{row["actor2"]}</span>',unsafe_allow_html=True)
        rc[3].markdown(f'<span style="font-size:12px;color:#0A0A0A">{row["description"]}</span>',unsafe_allow_html=True)
        rc[4].markdown(f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;font-weight:600;color:{gcol}">{sign}{g:.1f}</span>',unsafe_allow_html=True)
        st.markdown('<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown(
    '<div style="margin-top:64px;padding-top:20px;'
    'border-top:1px solid #E5E5E5;display:flex;'
    'justify-content:space-between;font-family:IBM Plex Mono,monospace;'
    'font-size:11px;color:#9B9B9B">'
    '<span>ARES · Adaptive Relationship &amp; Event Simulator</span>'
    '<span>World Bank · Guardian API · GDELT · UN Voting Records</span>'
    '<span>IIM Calcutta · SNA Group 5 · MBA Batch 62 · 2026</span>'
    '</div>',
    unsafe_allow_html=True)
