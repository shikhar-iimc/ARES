import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle

st.set_page_config(
    page_title="ARES — Conflict Intelligence",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── INJECT STYLES via st.html (bypasses sanitizer) ────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background: #FFFFFF !important; }
.block-container { padding: 0 40px 80px 40px !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"], section[data-testid="stSidebar"] { display: none !important; }
.top-bar { position:fixed; top:0; left:0; right:0; height:3px; background:#C41E3A; z-index:9999; }

/* global font */
html, body, p, div, span, h1, h2, h3, label, input, button,
[class*="css"], .stMarkdown, .stText, .element-container {
  font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
  color: #0A0A0A !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important;
  background: transparent !important;
  border-bottom: 1px solid #E5E5E5 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  color: #333333 !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 12px 22px !important;
  text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
  color: #0A0A0A !important;
  border-bottom: 2px solid #C41E3A !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* checkboxes */
.stCheckbox label, .stCheckbox label p,
.stCheckbox span[data-testid="stMarkdownContainer"] p {
  color: #0A0A0A !important;
  font-size: 13px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 400 !important;
}

/* radio */
.stRadio label, .stRadio label p,
.stRadio div[data-testid="stMarkdownContainer"] p {
  color: #0A0A0A !important;
  font-size: 13px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

/* slider label */
.stSlider label, .stSlider label p { color: #0A0A0A !important; }

/* text input label */
.stTextInput label, .stTextInput label p { color: #0A0A0A !important; }
.stTextInput input {
  border: 1px solid #E5E5E5 !important;
  border-radius: 0 !important;
  background: #FFFFFF !important;
  color: #0A0A0A !important;
  font-size: 13px !important;
}

/* metric */
[data-testid="metric-container"] {
  background: #FFFFFF !important;
  border: 1px solid #E5E5E5 !important;
  border-radius: 0 !important;
  padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: #6B6B6B !important;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 26px !important;
  font-weight: 600 !important;
  color: #0A0A0A !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; color: #6B6B6B !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }
</style>
<div class="top-bar"></div>
""")

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("ares_data.pkl", "rb") as f:
        return pickle.load(f)

data      = load_data()
SIM_LOG   = data["sim_log"]
G_INITIAL = data["g_initial"]
G_FINAL   = data["g_final"]
EVENTS_DF = data["events_df"]

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

# ── HELPERS ───────────────────────────────────────────────────
def sec(label):
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:#6B6B6B;margin:28px 0 12px 0;'
        f'padding-bottom:8px;border-bottom:1px solid #EEEEEE">{label}</div>',
        unsafe_allow_html=True
    )

def mono(txt, color="#6B6B6B", size="12px"):
    return (f'<span style="font-family:IBM Plex Mono,monospace;'
            f'font-size:{size};color:{color}">{txt}</span>')

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
        unsafe_allow_html=True
    )
with h2:
    st.markdown(
        '<div style="padding:44px 0 20px 0;border-bottom:1px solid #E5E5E5;'
        'text-align:right">'
        '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        'color:#9B9B9B;line-height:2.2;letter-spacing:0.06em">'
        'IIM CALCUTTA<br>SNA · GROUP 5<br>MBA BATCH 62</div></div>',
        unsafe_allow_html=True
    )

# ── TABS ──────────────────────────────────────────────────────
T = st.tabs(["OVERVIEW", "NETWORK", "TRAJECTORIES",
             "MEDIATOR ANALYSIS", "EVENT LOG"])

# ═══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════
with T[0]:
    n_ev  = len(EVENTS_DF)
    n_h   = int((EVENTS_DF["goldstein"] < 0).sum())
    n_d   = int((EVENTS_DF["goldstein"] > 0).sum())
    ir_s  = float(SIM_LOG.iloc[0]["iran_might"])
    ir_e  = float(SIM_LOG.iloc[-1]["iran_might"])
    ir_l  = (1 - ir_e/ir_s)*100 if ir_s > 0 else 0
    fmed  = SIM_LOG.iloc[-1]["top_mediator"]

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Events",       str(n_ev), "Oct 2023 – Aug 2026")
    m2.metric("Hostile Events",     str(n_h),  f"{n_h/n_ev*100:.0f}% of total")
    m3.metric("Diplomatic Events",  str(n_d),  f"{n_d/n_ev*100:.0f}% of total")
    m4.metric("Iran Might Loss",    f"{ir_l:.0f}%", f"{ir_s:.3f} → {ir_e:.3f}")
    m5.metric("Structural Mediator", fmed,     "by network position")

    sec("Conflict Phases")
    phases = [
        ("PHASE I",   "Oct 2023 – Dec 2023",
         "Hamas Oct 7 attack triggers regional cascade. Houthis and Hezbollah activate. US expands Iran sanctions."),
        ("PHASE II",  "Jan 2024 – Oct 2024",
         "First direct Iran–Israel military exchanges. 300+ drones and missiles. Nasrallah and Sinwar killed."),
        ("PHASE III", "Nov 2024 – Feb 2026",
         "Lebanon ceasefire. Assad falls. Twelve-Day War (Jun 2025): US joins Israel strikes on Iran nuclear sites."),
        ("PHASE IV",  "Feb 2026 – Present",
         "Operation Epic Fury: 900 strikes, Khamenei killed. Strait of Hormuz crisis. Pakistan brokers ceasefire. MOU signed."),
    ]
    for pid, dates, desc in phases:
        c1, c2, c3 = st.columns([1, 2, 6])
        c1.markdown(mono(pid, "#C41E3A", "11px"), unsafe_allow_html=True)
        c2.markdown(mono(dates, "#6B6B6B", "12px"), unsafe_allow_html=True)
        c3.markdown(f'<span style="font-size:13px;color:#0A0A0A">{desc}</span>',
                    unsafe_allow_html=True)
        st.markdown('<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',
                    unsafe_allow_html=True)

    sec("National Power Trajectory")
    fig_m = go.Figure()
    for ctry, col, dash in [
        ("Iran",   "#C41E3A", "solid"),
        ("Israel", "#1B3A6B", "solid"),
        ("USA",    "#1B7A3E", "dash"),
    ]:
        cn = f"{ctry.lower()}_might"
        if cn in SIM_LOG.columns:
            fig_m.add_trace(go.Scatter(
                x=SIM_LOG["date"], y=SIM_LOG[cn], name=ctry,
                mode="lines", line=dict(color=col, width=2, dash=dash)
            ))
    for d, l in [("2024-04-13","Direct exchange"),
                 ("2025-06-13","12-Day War"),
                 ("2026-02-28","Epic Fury"),
                 ("2026-04-08","Ceasefire")]:
        fig_m.add_vline(x=d, line_dash="dot", line_color="#CCCCCC",
                        line_width=1, annotation_text=l,
                        annotation_textangle=-90,
                        annotation_font=dict(size=9, color="#999999"))
    fig_m.update_layout(**PB, height=260)
    fig_m.update_xaxes(**AX, title_text="")
    fig_m.update_yaxes(**AX, title_text="Might [0–1]", range=[0, 1.05])
    st.plotly_chart(fig_m, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2 — NETWORK
# ═══════════════════════════════════════════════
with T[1]:
    cc, cm = st.columns([1, 4])
    with cc:
        sec("Controls")
        net_ch = st.radio(
            "Network state",
            ["Initial — Oct 2023", "Final — Aug 2026"],
            label_visibility="collapsed"
        )
        thresh = st.slider("Edge threshold", 0.0, 1.0, 0.25, 0.05)
        st.markdown("<br>", unsafe_allow_html=True)
        for dot_col, lbl in [("#1B3A6B","US-aligned"),
                              ("#C41E3A","Iran-aligned"),
                              ("#888888","Straddler")]:
            st.markdown(
                f'<div style="font-size:12px;color:#0A0A0A;margin-bottom:6px">'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{dot_col};margin-right:8px;'
                f'vertical-align:middle"></span>{lbl}</div>',
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)
        for line_col, lbl in [("#1B7A3E","Friendly"),
                               ("#C41E3A","Hostile")]:
            st.markdown(
                f'<div style="font-size:12px;color:#0A0A0A;margin-bottom:6px">'
                f'<span style="color:{line_col};font-weight:700;'
                f'margin-right:8px">—</span>{lbl}</div>',
                unsafe_allow_html=True
            )
        st.markdown('<div style="font-size:12px;color:#6B6B6B;margin-top:8px">'
                    'Node size = Might</div>', unsafe_allow_html=True)

    with cm:
        G_sh = G_INITIAL if "Initial" in net_ch else G_FINAL
        ns_s = sorted(G_sh.nodes(data=True),
                      key=lambda x: x[1].get("alignment", 0), reverse=True)
        nn   = len(ns_s)
        pos  = {}
        for i, (nd, _) in enumerate(ns_s):
            a = 2 * np.pi * i / nn
            pos[nd] = (np.cos(a), np.sin(a))

        etrs = []
        for a, b, ed in G_sh.edges(data=True):
            r = ed.get("relationship", 0)
            if abs(r) < thresh:
                continue
            x0, y0 = pos[a]; x1, y1 = pos[b]
            op  = min(0.9, 0.2 + abs(r) * 0.7)
            col = (f"rgba(27,122,62,{op})" if r >= 0
                   else f"rgba(196,30,58,{op})")
            etrs.append(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=max(0.5, abs(r)*2.5), color=col),
                hoverinfo="none", showlegend=False
            ))

        nx_l,ny_l,nt_l,nh_l,nc_l,nz_l = [],[],[],[],[],[]
        for nd, nd_d in G_sh.nodes(data=True):
            x, y = pos[nd]
            nx_l.append(x); ny_l.append(y); nt_l.append(nd)
            al = nd_d.get("alignment", 0)
            m  = nd_d.get("might", 0.1)
            camp = "US" if al>0.15 else ("Iran" if al<-0.15 else "Neutral")
            nh_l.append(f"<b>{nd}</b><br>Might:{m:.3f}<br>"
                        f"Alignment:{al:+.3f}<br>Camp:{camp}")
            nc_l.append("#1B3A6B" if al>0.15 else
                        ("#C41E3A" if al<-0.15 else "#888888"))
            nz_l.append(12 + 35*float(m))

        ntr = go.Scatter(
            x=nx_l, y=ny_l, mode="markers+text",
            text=nt_l, textposition="top center",
            textfont=dict(family="IBM Plex Sans", size=10, color="#0A0A0A"),
            hovertext=nh_l, hoverinfo="text",
            marker=dict(color=nc_l, size=nz_l,
                        line=dict(width=1.5, color="#FFFFFF")),
            showlegend=False
        )
        lbl = ("Initial Network — Oct 2023"
               if "Initial" in net_ch else "Final Network — Aug 2026")
        fig_n = go.Figure(data=etrs + [ntr])
        fig_n.update_layout(
            height=580, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            margin=dict(l=0,r=0,t=36,b=0),
            font=dict(family="IBM Plex Sans"),
            title=dict(text=lbl,
                       font=dict(size=13,color="#6B6B6B",
                                 family="IBM Plex Sans"), x=0)
        )
        fig_n.update_xaxes(showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="y")
        fig_n.update_yaxes(showgrid=False, zeroline=False,
                           showticklabels=False)
        st.plotly_chart(fig_n, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3 — TRAJECTORIES
# ═══════════════════════════════════════════════
with T[2]:
    sec("Network Tension Over Time")
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
    for d, l in [("2024-04-13","First direct exchange"),
                 ("2025-06-13","Twelve-Day War"),
                 ("2026-02-28","Operation Epic Fury"),
                 ("2026-04-08","Pakistan ceasefire")]:
        fig_t.add_vline(x=d, line_dash="dot", line_color="#CCCCCC",
                        line_width=1, annotation_text=l,
                        annotation_textangle=-90,
                        annotation_font=dict(size=9, color="#999"))
    fig_t.update_layout(**PB, height=340)
    fig_t.update_xaxes(**AX, title_text="")
    fig_t.update_yaxes(**AX, title_text="Value")
    st.plotly_chart(fig_t, use_container_width=True)

    sec("Mediator Leadership")
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
    fig_b.update_xaxes(**AX, title_text="Event count")
    fig_b.update_yaxes(**AX, title_text="")
    st.plotly_chart(fig_b, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4 — MEDIATOR ANALYSIS
# ═══════════════════════════════════════════════
with T[3]:

    # ── playback state ──
    if "tick" not in st.session_state:
        st.session_state.tick = len(SIM_LOG) - 1
    if "playing" not in st.session_state:
        st.session_state.playing = False

    # ── controls row ──
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 6])
    with ctrl1:
        if st.button("▶ Play" if not st.session_state.playing
                     else "⏸ Pause"):
            st.session_state.playing = not st.session_state.playing
            if st.session_state.playing:
                st.session_state.tick = 0
    with ctrl2:
        if st.button("↺ Reset"):
            st.session_state.tick = 0
            st.session_state.playing = False

    # manual slider (disabled during playback)
    tick_idx = st.slider(
        "Event index",
        min_value=0,
        max_value=len(SIM_LOG) - 1,
        value=st.session_state.tick,
        disabled=st.session_state.playing,
        key="manual_slider"
    )
    if not st.session_state.playing:
        st.session_state.tick = tick_idx

    # ── current snapshot ──
    sel     = SIM_LOG.iloc[st.session_state.tick]
    top3    = sel["top3"]
    snap_dt = sel["date"]
    snap_ev = sel["event"]

    # progress bar
    pct = int((st.session_state.tick / (len(SIM_LOG)-1)) * 100)
    st.markdown(
        f'<div style="background:#F0F0F0;height:3px;width:100%;margin-bottom:20px">'
        f'<div style="background:#C41E3A;height:3px;width:{pct}%"></div></div>',
        unsafe_allow_html=True
    )

    # date + event label
    sec(f"Mediator Ranking · {snap_dt}")
    st.markdown(
        f'<div style="font-size:12px;color:#6B6B6B;margin-bottom:20px">'
        f'{snap_ev}</div>',
        unsafe_allow_html=True
    )

    # ── podium circles ──
    if len(top3) >= 3:
        p1, p2, p3 = st.columns(3)
        for col_w, (name, score), border, sz, rnk in [
            (p1, top3[0], "#1B3A6B", "120px", "RANK 01"),
            (p2, top3[1], "#888888", "100px", "RANK 02"),
            (p3, top3[2], "#C8A87A", "100px", "RANK 03"),
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
                    unsafe_allow_html=True
                )

    # ── mediator score bars (all candidates at this tick) ──
    sec("All Mediator Scores at This Moment")
    all_scores = sorted(top3, key=lambda x: x[1], reverse=True)
    max_score  = max(s for _,s in all_scores) if all_scores else 1
    for name, score in all_scores:
        bw = int((score / max_score) * 300) if max_score > 0 else 0
        col = "#C41E3A" if name == top3[0][0] else "#1B3A6B"
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
            unsafe_allow_html=True
        )

    # ── key moments table ──
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
    for hc, hl in zip(hcols, ["Date","Mediator","Score","Event"]):
        hc.markdown(
            f'<span style="font-size:10px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#6B6B6B">{hl}</span>',
            unsafe_allow_html=True
        )
    st.markdown(
        '<hr style="margin:4px 0 0 0;border:none;border-top:1px solid #E5E5E5">',
        unsafe_allow_html=True
    )
    for d, label in key_moments:
        match = SIM_LOG[SIM_LOG["date"] >= d]
        if not match.empty:
            row = match.iloc[0]
            # highlight if this is the current tick's date
            is_now = (snap_dt >= d and
                      (key_moments.index((d,label)) == len(key_moments)-1 or
                       snap_dt < key_moments[key_moments.index((d,label))+1][0]
                       if key_moments.index((d,label)) < len(key_moments)-1
                       else True))
            bg = "background:#F0F8FF;" if is_now else ""
            rc = st.columns([2,2,1,4])
            rc[0].markdown(
                mono(d, "#C41E3A" if is_now else "#6B6B6B", "12px"),
                unsafe_allow_html=True
            )
            rc[1].markdown(
                f'<span style="font-weight:600;font-size:13px;color:#1B3A6B">'
                f'{row["top_mediator"]}</span>',
                unsafe_allow_html=True
            )
            rc[2].markdown(
                mono(f'{row["top_score"]:.4f}', "#888888", "12px"),
                unsafe_allow_html=True
            )
            rc[3].markdown(
                f'<span style="font-size:12px;color:#0A0A0A">{label}</span>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',
            unsafe_allow_html=True
        )

    # finding + model vs reality (unchanged)
    st.markdown(
        '<div style="background:#F8F8F8;border-left:3px solid #1B3A6B;'
        'padding:16px 20px;margin:24px 0;font-size:13px;line-height:1.7;'
        'color:#0A0A0A">'
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.15em;color:#1B3A6B;margin-bottom:8px">Key Finding</div>'
        'Our structural model identifies <b>India and China</b> as best-positioned '
        'mediators by network topology. Real-world mediation was carried out by '
        '<b>Qatar</b>, <b>Oman</b>, and <b>Pakistan</b>. This divergence shows '
        'structural position is <em>necessary but not sufficient</em> for mediation.'
        '</div>',
        unsafe_allow_html=True
    )

    col_mod, col_real = st.columns(2)
    with col_mod:
        sec("Model Output")
        for rank, (name, score) in enumerate(SIM_LOG.iloc[-1]["top3"], 1):
            bw = min(int(score * 900), 300)
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
                unsafe_allow_html=True
            )
    with col_real:
        sec("Real-World Mediators")
        for name, roles in [
            ("Qatar",    "Nov 2023 hostage deal · Jan 2025 Gaza ceasefire"),
            ("Oman",     "2025 US–Iran back-channel · Strait of Hormuz MOU"),
            ("Pakistan", "Apr 2026 ceasefire · Jun 2026 14-point MOU"),
        ]:
            st.markdown(
                f'<div style="margin-bottom:16px">'
                f'<div style="font-size:14px;font-weight:600;color:#0A0A0A;'
                f'margin-bottom:3px">{name}</div>'
                f'<div style="font-size:12px;color:#6B6B6B;line-height:1.6">'
                f'{roles}</div></div>',
                unsafe_allow_html=True
            )

    # ── AUTO-ADVANCE (must be last in tab) ──
    if st.session_state.playing:
        import time
        time.sleep(0.4)   # 0.4s per tick → full playback ~20 seconds
        if st.session_state.tick < len(SIM_LOG) - 1:
            st.session_state.tick += 1
        else:
            st.session_state.playing = False
        st.rerun()

# ═══════════════════════════════════════════════
# TAB 5 — EVENT LOG
# ═══════════════════════════════════════════════
with T[4]:
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

    sec(f"{len(filt)} Events")

    fig_g = go.Figure()
    fig_g.add_bar(
        x=filt["date"].astype(str), y=filt["goldstein"],
        marker_color=["#C41E3A" if g<0 else "#1B7A3E"
                      for g in filt["goldstein"]],
        hovertext=(filt["actor1"]+" → "+filt["actor2"]
                   +"<br>"+filt["description"]),
        hoverinfo="text+y"
    )
    fig_g.update_layout(**PB, height=220, showlegend=False, bargap=0.15)
    fig_g.update_xaxes(**AX, title_text="")
    fig_g.update_yaxes(**AX, title_text="Goldstein Scale")
    st.plotly_chart(fig_g, use_container_width=True)

    # table header
    hcols2 = st.columns([2, 1.5, 1.5, 4, 1])
    for hc, hl in zip(hcols2, ["Date","Actor 1","Actor 2","Description","Score"]):
        hc.markdown(
            f'<span style="font-size:10px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#6B6B6B">{hl}</span>',
            unsafe_allow_html=True
        )
    st.markdown('<hr style="margin:4px 0 0 0;border:none;'
                'border-top:2px solid #E5E5E5">', unsafe_allow_html=True)

    for _, row in filt.iterrows():
        g    = row["goldstein"]
        sign = "+" if g >= 0 else ""
        gcol = "#C41E3A" if g < 0 else "#1B7A3E"
        rc   = st.columns([2, 1.5, 1.5, 4, 1])
        rc[0].markdown(mono(str(row["date"].date()), "#6B6B6B", "12px"),
                       unsafe_allow_html=True)
        rc[1].markdown(
            f'<span style="font-weight:500;font-size:12px;color:#0A0A0A">'
            f'{row["actor1"]}</span>', unsafe_allow_html=True)
        rc[2].markdown(
            f'<span style="font-weight:500;font-size:12px;color:#0A0A0A">'
            f'{row["actor2"]}</span>', unsafe_allow_html=True)
        rc[3].markdown(
            f'<span style="font-size:12px;color:#0A0A0A">'
            f'{row["description"]}</span>', unsafe_allow_html=True)
        rc[4].markdown(
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;'
            f'font-weight:600;color:{gcol}">{sign}{g:.1f}</span>',
            unsafe_allow_html=True)
        st.markdown('<hr style="margin:0;border:none;border-top:1px solid #F5F5F5">',
                    unsafe_allow_html=True)

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
    unsafe_allow_html=True
)
