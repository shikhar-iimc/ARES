import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import pickle

st.set_page_config(
    page_title="ARES — Conflict Intelligence",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: #FFFFFF !important;
    color: #0A0A0A !important;
  }
  .top-bar {
    position:fixed; top:0; left:0; right:0;
    height:3px; background:#C41E3A; z-index:9999;
  }
  /* hide streamlit chrome */
  #MainMenu, footer, header,
  [data-testid="collapsedControl"],
  section[data-testid="stSidebar"] { display:none !important; }
  .block-container { padding:0 48px 64px 48px !important; max-width:1400px; }

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap:0; border-bottom:1px solid #E5E5E5;
    background:transparent !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family:'IBM Plex Sans',sans-serif !important;
    font-size:12px !important; font-weight:500 !important;
    letter-spacing:0.08em !important;
    color:#6B6B6B !important;
    background:transparent !important;
    border:none !important; border-radius:0 !important;
    padding:12px 20px !important;
  }
  .stTabs [aria-selected="true"] {
    color:#0A0A0A !important;
    border-bottom:2px solid #C41E3A !important;
    background:transparent !important;
  }
  .stTabs [data-baseweb="tab-highlight"] { display:none !important; }
  .stTabs [data-baseweb="tab-border"]    { display:none !important; }

  /* inputs */
  .stTextInput input {
    font-family:'IBM Plex Sans',sans-serif !important;
    border:1px solid #E5E5E5 !important;
    border-radius:0 !important;
    font-size:13px !important;
    background:#FFFFFF !important;
    color:#0A0A0A !important;
  }
  .stCheckbox label {
    font-family:'IBM Plex Sans',sans-serif !important;
    font-size:13px !important; color:#0A0A0A !important;
  }
  .stSlider [data-testid="stThumbValue"] {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:11px !important;
  }
  div[data-baseweb="slider"] div[role="slider"] {
    background:#C41E3A !important;
    border-color:#C41E3A !important;
  }
  .stRadio label {
    font-family:'IBM Plex Sans',sans-serif !important;
    font-size:13px !important; color:#0A0A0A !important;
  }
  /* metric cards */
  [data-testid="metric-container"] {
    background:#FFFFFF !important;
    border:1px solid #E5E5E5 !important;
    border-radius:0 !important; padding:20px 24px !important;
  }
  [data-testid="metric-container"] label {
    font-family:'IBM Plex Sans',sans-serif !important;
    font-size:11px !important; font-weight:600 !important;
    text-transform:uppercase !important;
    letter-spacing:0.1em !important; color:#6B6B6B !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family:'IBM Plex Mono',monospace !important;
    font-size:28px !important; font-weight:600 !important;
    color:#0A0A0A !important;
  }
  /* section label */
  .sec-label {
    font-size:11px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.12em;
    color:#6B6B6B; margin:24px 0 12px 0;
    padding-bottom:8px; border-bottom:1px solid #F0F0F0;
  }
  /* phase rows */
  .ph-row {
    display:grid; grid-template-columns:90px 150px 1fr;
    gap:16px; padding:13px 0;
    border-bottom:1px solid #F5F5F5; align-items:start;
  }
  .ph-id {
    font-family:'IBM Plex Mono',monospace;
    font-size:11px; font-weight:500; color:#C41E3A;
  }
  .ph-date { font-size:12px; color:#6B6B6B; }
  .ph-desc { font-size:13px; color:#0A0A0A; line-height:1.5; }
  /* finding */
  .finding {
    background:#F8F8F8; border-left:3px solid #1B3A6B;
    padding:16px 20px; margin:20px 0;
    font-size:13px; line-height:1.65; color:#0A0A0A;
  }
  .finding-lbl {
    font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.15em; color:#1B3A6B; margin-bottom:6px;
  }
  /* podium circles */
  .podium-wrap {
    display:flex; gap:24px; justify-content:center;
    margin:8px 0 28px 0;
  }
  .pod {
    display:flex; flex-direction:column; align-items:center;
    gap:8px;
  }
  .pod-circle {
    width:96px; height:96px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    flex-direction:column;
    border:2px solid #E5E5E5;
    background:#FFFFFF;
  }
  .pod-circle.rank1 { border-color:#1B3A6B; width:112px; height:112px; }
  .pod-circle.rank2 { border-color:#888888; }
  .pod-circle.rank3 { border-color:#C8A87A; }
  .pod-name {
    font-size:13px; font-weight:700; color:#0A0A0A;
    text-align:center; line-height:1.2;
  }
  .pod-score {
    font-family:'IBM Plex Mono',monospace;
    font-size:10px; color:#6B6B6B;
  }
  .pod-rank {
    font-family:'IBM Plex Mono',monospace;
    font-size:10px; color:#9B9B9B; letter-spacing:0.08em;
  }
  /* moment rows */
  .mom-row {
    display:grid; grid-template-columns:110px 110px 70px 1fr;
    gap:12px; padding:9px 0;
    border-bottom:1px solid #F5F5F5;
    font-size:12px; align-items:center;
  }
  .mom-date { font-family:'IBM Plex Mono',monospace; color:#6B6B6B; }
  .mom-med  { font-weight:600; color:#1B3A6B; }
  .mom-sc   { font-family:'IBM Plex Mono',monospace; color:#888888; }
  .mom-lbl  { color:#0A0A0A; }
  /* bar rows */
  .bar-row { margin-bottom:14px; }
  .bar-meta {
    display:flex; justify-content:space-between;
    font-size:12px; margin-bottom:4px;
  }
  .bar-meta b { color:#0A0A0A; }
  .bar-meta span { font-family:'IBM Plex Mono',monospace; color:#6B6B6B; }
  .bar-track { background:#F0F0F0; height:3px; width:100%; }
  .bar-fill  { background:#1B3A6B; height:3px; }
  /* real mediator cards */
  .rm-card { margin-bottom:16px; }
  .rm-name { font-size:14px; font-weight:600; color:#0A0A0A; margin-bottom:3px; }
  .rm-role { font-size:12px; color:#6B6B6B; line-height:1.5; }
  /* event log */
  .ev-hdr {
    display:grid; grid-template-columns:100px 90px 90px 1fr 60px;
    gap:12px; padding:8px 0; border-bottom:2px solid #E5E5E5;
    font-size:10px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#6B6B6B;
  }
  .ev-row {
    display:grid; grid-template-columns:100px 90px 90px 1fr 60px;
    gap:12px; padding:9px 0; border-bottom:1px solid #F5F5F5;
    font-size:12px; align-items:center;
  }
  .ev-date { font-family:'IBM Plex Mono',monospace; color:#6B6B6B; }
  .ev-act  { font-weight:500; color:#0A0A0A; }
  .ev-desc { color:#0A0A0A; }
  .ev-sc   { font-family:'IBM Plex Mono',monospace; font-weight:600; text-align:right; }
  .ev-sc.h { color:#C41E3A; }
  .ev-sc.d { color:#1B7A3E; }
  /* network legend */
  .leg { font-size:12px; color:#6B6B6B; line-height:2.2; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
  /* footer */
  .ares-footer {
    margin-top:64px; padding-top:20px;
    border-top:1px solid #E5E5E5;
    display:flex; justify-content:space-between;
    font-family:'IBM Plex Mono',monospace;
    font-size:11px; color:#9B9B9B;
  }
</style>
<div class="top-bar"></div>
""", unsafe_allow_html=True)

# ── LOAD ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("ares_data.pkl", "rb") as f:
        return pickle.load(f)

data      = load_data()
SIM_LOG   = data["sim_log"]
G_INITIAL = data["g_initial"]
G_FINAL   = data["g_final"]
EVENTS_DF = data["events_df"]

PLOT_BASE = dict(
    font=dict(family="IBM Plex Sans", size=12, color="#0A0A0A"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    margin=dict(l=0, r=0, t=24, b=0),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="left", x=0,
                font=dict(size=11))
)
AX = dict(gridcolor="#F0F0F0", linecolor="#E5E5E5",
          tickfont=dict(family="IBM Plex Mono", size=11))

# ── HEADER ────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""
    <div style="padding:40px 0 24px 0; border-bottom:1px solid #E5E5E5; margin-bottom:0">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;
                  font-weight:500;letter-spacing:0.25em;color:#C41E3A;margin-bottom:4px">
        ▲ ARES
      </div>
      <div style="font-size:28px;font-weight:700;letter-spacing:-0.5px;
                  color:#0A0A0A;line-height:1.1;margin-bottom:6px">
        Adaptive Relationship &amp; Event Simulator
      </div>
      <div style="font-size:13px;color:#6B6B6B">
        US–Iran–Israel Conflict Network &nbsp;·&nbsp;
        Oct 2023 → Aug 2026 &nbsp;·&nbsp;
        Structural Balance Theory (Marvel–Kleinberg–Strogatz 2011)
      </div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="padding:40px 0 24px 0;border-bottom:1px solid #E5E5E5;text-align:right">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                  color:#9B9B9B;letter-spacing:0.05em;line-height:2">
        IIM CALCUTTA<br>SNA · GROUP 5<br>MBA BATCH 62
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tabs = st.tabs(["OVERVIEW","NETWORK","TRAJECTORIES",
                "MEDIATOR ANALYSIS","EVENT LOG"])

# ═══════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════
with tabs[0]:
    hostile_n = int((EVENTS_DF["goldstein"] < 0).sum())
    diplo_n   = int((EVENTS_DF["goldstein"] > 0).sum())
    iran_s    = float(SIM_LOG.iloc[0]["iran_might"])
    iran_e    = float(SIM_LOG.iloc[-1]["iran_might"])
    iran_loss = (1 - iran_e/iran_s)*100 if iran_s > 0 else 0
    final_med = SIM_LOG.iloc[-1]["top_mediator"]

    mc = st.columns(5)
    mc[0].metric("Total Events",      str(len(EVENTS_DF)), "Oct 2023 – Aug 2026")
    mc[1].metric("Hostile Events",    str(hostile_n),      f"{hostile_n/len(EVENTS_DF)*100:.0f}% of total")
    mc[2].metric("Diplomatic Events", str(diplo_n),        f"{diplo_n/len(EVENTS_DF)*100:.0f}% of total")
    mc[3].metric("Iran Might Loss",   f"{iran_loss:.0f}%", f"{iran_s:.3f} → {iran_e:.3f}")
    mc[4].metric("Structural Mediator", final_med,         "by network position")

    st.markdown('<div class="sec-label">Conflict Phases</div>', unsafe_allow_html=True)
    phases = [
        ("PHASE I",   "Oct 2023 – Dec 2023",
         "Hamas Oct 7 attack triggers regional cascade. Houthis and Hezbollah activate. US expands Iran sanctions."),
        ("PHASE II",  "Jan 2024 – Oct 2024",
         "First direct Iran–Israel military exchanges. 300+ drones and missiles. Nasrallah and Sinwar killed."),
        ("PHASE III", "Nov 2024 – Feb 2026",
         "Lebanon ceasefire. Assad regime falls. Twelve-Day War (Jun 2025): US joins Israel strikes on Iran nuclear sites."),
        ("PHASE IV",  "Feb 2026 – Present",
         "Operation Epic Fury: 900 strikes, Khamenei killed. Strait of Hormuz crisis. Pakistan brokers ceasefire. MOU signed."),
    ]
    for pid, dates, desc in phases:
        st.markdown(
            f'<div class="ph-row">'
            f'<div class="ph-id">{pid}</div>'
            f'<div class="ph-date">{dates}</div>'
            f'<div class="ph-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sec-label" style="margin-top:28px">National Power Trajectory</div>',
                unsafe_allow_html=True)
    fig_m = go.Figure()
    for ctry, col, dash in [
        ("Iran","#C41E3A","solid"),
        ("Israel","#1B3A6B","solid"),
        ("USA","#1B7A3E","dash")
    ]:
        cn = f"{ctry.lower()}_might"
        if cn in SIM_LOG.columns:
            fig_m.add_trace(go.Scatter(
                x=SIM_LOG["date"], y=SIM_LOG[cn],
                name=ctry, mode="lines",
                line=dict(color=col, width=2, dash=dash)
            ))
    for d,l in [("2024-04-13","Direct exchange"),
                ("2025-06-13","12-Day War"),
                ("2026-02-28","Epic Fury"),
                ("2026-04-08","Ceasefire")]:
        fig_m.add_vline(x=d, line_dash="dot",
                        line_color="#CCCCCC", line_width=1,
                        annotation_text=l,
                        annotation_textangle=-90,
                        annotation_font=dict(size=9, color="#999"))
    fig_m.update_layout(**PLOT_BASE, height=260)
    fig_m.update_xaxes(**AX, title_text="")
    fig_m.update_yaxes(**AX, title_text="Might [0–1]", range=[0,1.05])
    st.plotly_chart(fig_m, use_container_width=True)

# ═══════════════════════════════════════
# TAB 2 — NETWORK
# ═══════════════════════════════════════
with tabs[1]:
    cc, cm = st.columns([1,4])
    with cc:
        st.markdown('<div class="sec-label">Controls</div>', unsafe_allow_html=True)
        net_ch = st.radio("State",
                          ["Initial — Oct 2023","Final — Aug 2026"],
                          label_visibility="collapsed")
        thresh = st.slider("Edge threshold", 0.0, 1.0, 0.25, 0.05)
        st.markdown("""
        <div class="leg" style="margin-top:16px">
          <span class="dot" style="background:#1B3A6B"></span>US-aligned<br>
          <span class="dot" style="background:#C41E3A"></span>Iran-aligned<br>
          <span class="dot" style="background:#888888"></span>Straddler<br>
          <br>
          <span style="color:#1B7A3E;font-weight:600">—</span> Friendly<br>
          <span style="color:#C41E3A;font-weight:600">—</span> Hostile<br>
          <br>Node size = Might
        </div>
        """, unsafe_allow_html=True)

    with cm:
        G_sh = G_INITIAL if "Initial" in net_ch else G_FINAL
        ns = sorted(G_sh.nodes(data=True),
                    key=lambda x: x[1].get("alignment",0), reverse=True)
        nn = len(ns)
        pos = {}
        for i,(nd,_) in enumerate(ns):
            a = 2*np.pi*i/nn
            pos[nd] = (np.cos(a), np.sin(a))

        etrs = []
        for a,b,ed in G_sh.edges(data=True):
            r = ed.get("relationship",0)
            if abs(r) < thresh: continue
            x0,y0=pos[a]; x1,y1=pos[b]
            op = min(0.9, 0.2+abs(r)*0.7)
            col = (f"rgba(27,122,62,{op})" if r>=0
                   else f"rgba(196,30,58,{op})")
            etrs.append(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=max(0.5,abs(r)*2.5), color=col),
                hoverinfo="none", showlegend=False
            ))

        nx_l, ny_l, nt_l, nh_l, nc_l, nz_l = [],[],[],[],[],[]
        for nd,nd_d in G_sh.nodes(data=True):
            x,y=pos[nd]; nx_l.append(x); ny_l.append(y)
            nt_l.append(nd)
            al=nd_d.get("alignment",0); m=nd_d.get("might",0.1)
            camp = "US" if al>0.15 else ("Iran" if al<-0.15 else "Neutral")
            nh_l.append(f"<b>{nd}</b><br>Might:{m:.3f}<br>Alignment:{al:+.3f}<br>Camp:{camp}")
            nc_l.append("#1B3A6B" if al>0.15 else ("#C41E3A" if al<-0.15 else "#888888"))
            nz_l.append(12+35*float(m))

        ntr = go.Scatter(
            x=nx_l, y=ny_l, mode="markers+text",
            text=nt_l, textposition="top center",
            textfont=dict(family="IBM Plex Sans", size=10, color="#0A0A0A"),
            hovertext=nh_l, hoverinfo="text",
            marker=dict(color=nc_l, size=nz_l,
                        line=dict(width=1.5, color="#FFFFFF")),
            showlegend=False
        )
        lbl = "Initial Network — Oct 2023" if "Initial" in net_ch else "Final Network — Aug 2026"
        fig_n = go.Figure(data=etrs+[ntr])
        fig_n.update_layout(height=580,
                            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                            margin=dict(l=0,r=0,t=32,b=0),
                            font=dict(family="IBM Plex Sans"),
                            title=dict(text=lbl,
                                       font=dict(size=13,color="#6B6B6B",
                                                 family="IBM Plex Sans"),x=0))
        fig_n.update_xaxes(showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="y")
        fig_n.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        st.plotly_chart(fig_n, use_container_width=True)

# ═══════════════════════════════════════
# TAB 3 — TRAJECTORIES
# ═══════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-label">Network Tension Over Time</div>', unsafe_allow_html=True)
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=SIM_LOG["date"], y=SIM_LOG["balance_energy"],
                               name="Balance Energy",
                               line=dict(color="#1B3A6B", width=2)))
    fig_t.add_trace(go.Scatter(x=SIM_LOG["date"], y=SIM_LOG["frac_unbalanced"],
                               name="Fraction Unbalanced",
                               line=dict(color="#C41E3A", width=2)))
    fig_t.add_trace(go.Scatter(x=SIM_LOG["date"], y=SIM_LOG["top_score"],
                               name="Top Mediator Score",
                               line=dict(color="#888888", width=1.5, dash="dot")))
    for d,l in [("2024-04-13","First direct exchange"),
                ("2025-06-13","Twelve-Day War"),
                ("2026-02-28","Operation Epic Fury"),
                ("2026-04-08","Pakistan ceasefire")]:
        fig_t.add_vline(x=d, line_dash="dot",
                        line_color="#CCCCCC", line_width=1,
                        annotation_text=l, annotation_textangle=-90,
                        annotation_font=dict(size=9, color="#999"))
    fig_t.update_layout(**PLOT_BASE, height=340)
    fig_t.update_xaxes(**AX, title_text="")
    fig_t.update_yaxes(**AX, title_text="Value")
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown('<div class="sec-label">Mediator Leadership</div>', unsafe_allow_html=True)
    mc2 = SIM_LOG["top_mediator"].value_counts().reset_index()
    mc2.columns = ["country","count"]
    mc2 = mc2.sort_values("count", ascending=True)
    top_c = mc2.iloc[-1]["country"]
    fig_b = go.Figure(go.Bar(
        x=mc2["count"], y=mc2["country"], orientation="h",
        marker=dict(color=["#C41E3A" if c==top_c else "#E5E5E5"
                           for c in mc2["country"]],
                    line=dict(width=0)),
        text=mc2["count"], textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11)
    ))
    fig_b.update_layout(**PLOT_BASE, height=320, showlegend=False)
    fig_b.update_xaxes(**AX, title_text="Events as top mediator")
    fig_b.update_yaxes(**AX, title_text="")
    st.plotly_chart(fig_b, use_container_width=True)

# ═══════════════════════════════════════
# TAB 4 — MEDIATOR ANALYSIS
# ═══════════════════════════════════════
with tabs[3]:
    cs, _ = st.columns([2,1])
    with cs:
        tick_idx = st.slider("Scroll through the conflict timeline",
                             0, len(SIM_LOG)-1, len(SIM_LOG)-1)

    sel      = SIM_LOG.iloc[tick_idx]
    top3     = sel["top3"]
    snap_dt  = sel["date"]
    snap_ev  = sel["event"]

    st.markdown(
        f'<div class="sec-label">Mediator Ranking · {snap_dt}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-size:12px;color:#6B6B6B;margin-bottom:20px">{snap_ev}</div>',
        unsafe_allow_html=True
    )

    # circles podium
    if len(top3) >= 3:
        rank_styles = [
            ("rank1","RANK 01","#1B3A6B"),
            ("rank2","RANK 02","#888888"),
            ("rank3","RANK 03","#C8A87A"),
        ]
        pod_html = '<div class="podium-wrap">'
        for i,(name,score) in enumerate(top3[:3]):
            cls,lbl,border = rank_styles[i]
            sz = "112px" if i==0 else "96px"
            pod_html += f"""
            <div class="pod">
              <div class="pod-circle {cls}"
                   style="width:{sz};height:{sz};border-color:{border}">
                <div class="pod-name">{name}</div>
                <div class="pod-score">{score:.4f}</div>
              </div>
              <div class="pod-rank">{lbl}</div>
            </div>
            """
        pod_html += "</div>"
        st.markdown(pod_html, unsafe_allow_html=True)

    # key moments — use st.columns to avoid raw HTML tables
    st.markdown('<div class="sec-label">Mediator at Key Moments</div>', unsafe_allow_html=True)
    key_moments = [
        ("2023-10-07","Oct 7 — Hamas attack"),
        ("2024-04-13","First direct Iran–Israel exchange"),
        ("2024-10-01","Iran fires 180 ballistic missiles"),
        ("2025-06-13","Twelve-Day War begins"),
        ("2026-02-28","Operation Epic Fury"),
        ("2026-04-08","Pakistan brokers ceasefire"),
    ]
    hdr = st.columns([2,2,1,4])
    hdr[0].markdown("**DATE**")
    hdr[1].markdown("**MEDIATOR**")
    hdr[2].markdown("**SCORE**")
    hdr[3].markdown("**EVENT**")
    for d,label in key_moments:
        match = SIM_LOG[SIM_LOG["date"] >= d]
        if not match.empty:
            row = match.iloc[0]
            rc  = st.columns([2,2,1,4])
            rc[0].markdown(
                f'<span style="font-family:IBM Plex Mono,monospace;'
                f'font-size:12px;color:#6B6B6B">{d}</span>',
                unsafe_allow_html=True
            )
            rc[1].markdown(
                f'<span style="font-weight:600;color:#1B3A6B;'
                f'font-size:13px">{row["top_mediator"]}</span>',
                unsafe_allow_html=True
            )
            rc[2].markdown(
                f'<span style="font-family:IBM Plex Mono,monospace;'
                f'font-size:12px;color:#888">{row["top_score"]:.4f}</span>',
                unsafe_allow_html=True
            )
            rc[3].markdown(
                f'<span style="font-size:12px;color:#0A0A0A">{label}</span>',
                unsafe_allow_html=True
            )

    # finding
    st.markdown("""
    <div class="finding">
      <div class="finding-lbl">Key Finding</div>
      Our structural model identifies <b>India and China</b> as best-positioned
      mediators by network topology — large states maintaining non-hostile ties
      to both camps throughout the conflict. Real-world mediation was carried
      out by <b>Qatar</b> (2023 hostage deal, 2025 Gaza ceasefire),
      <b>Oman</b> (US–Iran back-channel, Strait of Hormuz MOU), and
      <b>Pakistan</b> (Apr 2026 ceasefire, Jun 2026 14-point MOU). This
      divergence is itself a finding: structural position is
      <em>necessary but not sufficient</em> for mediation.
    </div>
    """, unsafe_allow_html=True)

    col_mod, col_real = st.columns(2)
    with col_mod:
        st.markdown('<div class="sec-label">Model Output</div>', unsafe_allow_html=True)
        final_top = SIM_LOG.iloc[-1]["top3"]
        for rank,(name,score) in enumerate(final_top,1):
            bw = min(int(score*800), 280)
            st.markdown(
                f'<div class="bar-row">'
                f'<div class="bar-meta"><b>#{rank} {name}</b>'
                f'<span>{score:.4f}</span></div>'
                f'<div class="bar-track">'
                f'<div class="bar-fill" style="width:{bw}px"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

    with col_real:
        st.markdown('<div class="sec-label">Real-World Mediators</div>', unsafe_allow_html=True)
        for name,roles in [
            ("Qatar",   "Nov 2023 hostage deal · Jan 2025 Gaza ceasefire"),
            ("Oman",    "2025 US–Iran back-channel · Strait of Hormuz MOU"),
            ("Pakistan","Apr 2026 ceasefire · Jun 2026 14-point MOU"),
        ]:
            st.markdown(
                f'<div class="rm-card">'
                f'<div class="rm-name">{name}</div>'
                f'<div class="rm-role">{roles}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

# ═══════════════════════════════════════
# TAB 5 — EVENT LOG
# ═══════════════════════════════════════
with tabs[4]:
    f1, f2, f3 = st.columns(3)
    with f1: show_h = st.checkbox("Hostile events", True)
    with f2: show_d = st.checkbox("Diplomatic events", True)
    with f3: srch   = st.text_input("Filter by actor", placeholder="e.g. Iran")

    filt = EVENTS_DF.copy()
    if not show_h: filt = filt[filt["goldstein"] >= 0]
    if not show_d: filt = filt[filt["goldstein"] <= 0]
    if srch:
        mask = (filt["actor1"].str.contains(srch, case=False, na=False) |
                filt["actor2"].str.contains(srch, case=False, na=False))
        filt = filt[mask]

    st.markdown(f'<div class="sec-label">{len(filt)} Events</div>',
                unsafe_allow_html=True)

    # goldstein chart
    fig_g = go.Figure()
    fig_g.add_bar(
        x=filt["date"].astype(str), y=filt["goldstein"],
        marker_color=["#C41E3A" if g<0 else "#1B7A3E"
                      for g in filt["goldstein"]],
        hovertext=filt["actor1"]+" → "+filt["actor2"]+"<br>"+filt["description"],
        hoverinfo="text+y"
    )
    fig_g.update_layout(**PLOT_BASE, height=220, showlegend=False, bargap=0.15)
    fig_g.update_xaxes(**AX, title_text="")
    fig_g.update_yaxes(**AX, title_text="Goldstein Scale")
    st.plotly_chart(fig_g, use_container_width=True)

    # event table using st.columns (avoids HTML rendering issue)
    hc = st.columns([2,1.5,1.5,4,1])
    for h,t in zip(hc,["DATE","ACTOR 1","ACTOR 2","DESCRIPTION","SCORE"]):
        h.markdown(
            f'<span style="font-size:10px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#6B6B6B">{t}</span>',
            unsafe_allow_html=True
        )
    st.markdown('<hr style="margin:4px 0 0 0;border:none;border-top:2px solid #E5E5E5">',
                unsafe_allow_html=True)

    for _, row in filt.iterrows():
        g    = row["goldstein"]
        sign = "+" if g >= 0 else ""
        gcol = "#C41E3A" if g < 0 else "#1B7A3E"
        rc   = st.columns([2,1.5,1.5,4,1])
        rc[0].markdown(
            f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:12px;color:#6B6B6B">{str(row["date"].date())}</span>',
            unsafe_allow_html=True
        )
        rc[1].markdown(
            f'<span style="font-weight:500;font-size:12px">{row["actor1"]}</span>',
            unsafe_allow_html=True
        )
        rc[2].markdown(
            f'<span style="font-weight:500;font-size:12px">{row["actor2"]}</span>',
            unsafe_allow_html=True
        )
        rc[3].markdown(
            f'<span style="font-size:12px;color:#0A0A0A">{row["description"]}</span>',
            unsafe_allow_html=True
        )
        rc[4].markdown(
            f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:12px;font-weight:600;color:{gcol}">'
            f'{sign}{g:.1f}</span>',
            unsafe_allow_html=True
        )

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="ares-footer">
  <span>ARES · Adaptive Relationship &amp; Event Simulator</span>
  <span>World Bank · Guardian API · GDELT · UN Voting Records</span>
  <span>IIM Calcutta · SNA Group 5 · MBA Batch 62 · 2026</span>
</div>
""", unsafe_allow_html=True)
