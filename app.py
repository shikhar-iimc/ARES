import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pickle
import os

st.set_page_config(
    page_title="ARES — Conflict Intelligence",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── TYPOGRAPHY + GLOBAL STYLES ────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        color: #0A0A0A;
    }
    .stApp { background: #FFFFFF; }
    .stApp > header { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .top-bar {
        position: fixed; top: 0; left: 0; right: 0;
        height: 3px; background: #C41E3A; z-index: 9999;
    }
    .ares-header {
        padding: 40px 0 24px 0;
        border-bottom: 1px solid #E5E5E5;
        margin-bottom: 0;
    }
    .ares-wordmark {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px; font-weight: 500;
        letter-spacing: 0.25em; color: #C41E3A;
        text-transform: uppercase; margin-bottom: 4px;
    }
    .ares-title {
        font-size: 28px; font-weight: 700;
        letter-spacing: -0.5px; color: #0A0A0A;
        line-height: 1.1; margin-bottom: 6px;
    }
    .ares-sub {
        font-size: 13px; color: #6B6B6B;
        font-weight: 400;
    }
    .ares-affil {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; color: #9B9B9B;
        text-align: right; letter-spacing: 0.05em;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1px; background: #E5E5E5;
        border: 1px solid #E5E5E5;
        margin-bottom: 32px;
    }
    .stat-card {
        background: #FFFFFF; padding: 20px 24px;
    }
    .stat-label {
        font-size: 11px; font-weight: 500;
        color: #6B6B6B; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 8px;
    }
    .stat-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px; font-weight: 600;
        color: #0A0A0A; line-height: 1;
    }
    .stat-value.threat { color: #C41E3A; }
    .stat-value.stable { color: #1B7A3E; }
    .stat-delta { font-size: 11px; color: #6B6B6B; margin-top: 4px; }
    .section-header {
        font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: #6B6B6B; margin-bottom: 16px;
        padding-bottom: 8px; border-bottom: 1px solid #F0F0F0;
    }
    .phase-row {
        display: grid;
        grid-template-columns: 80px 140px 1fr;
        gap: 16px; padding: 14px 0;
        border-bottom: 1px solid #F0F0F0;
        align-items: start;
    }
    .phase-id {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; font-weight: 500; color: #C41E3A;
    }
    .phase-date { font-size: 12px; color: #6B6B6B; }
    .phase-desc { font-size: 13px; color: #0A0A0A; line-height: 1.5; }
    .podium-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1px; background: #E5E5E5;
        border: 1px solid #E5E5E5; margin-bottom: 24px;
    }
    .podium-card {
        background: #FFFFFF; padding: 24px; text-align: center;
    }
    .podium-rank {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; color: #6B6B6B;
        letter-spacing: 0.1em; margin-bottom: 8px;
    }
    .podium-country {
        font-size: 22px; font-weight: 700;
        color: #0A0A0A; margin-bottom: 6px;
    }
    .podium-country.top { color: #1B3A6B; }
    .podium-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px; color: #6B6B6B;
    }
    .event-row {
        display: grid;
        grid-template-columns: 100px 90px 100px 1fr 60px;
        gap: 12px; padding: 10px 0;
        border-bottom: 1px solid #F5F5F5;
        align-items: center; font-size: 12px;
    }
    .event-date {
        color: #6B6B6B;
        font-family: 'IBM Plex Mono', monospace;
    }
    .event-actor { font-weight: 500; color: #0A0A0A; }
    .event-desc { color: #0A0A0A; }
    .event-score {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600; text-align: right;
    }
    .event-score.hostile { color: #C41E3A; }
    .event-score.diplo   { color: #1B7A3E; }
    .finding-box {
        background: #F8F8F8;
        border-left: 3px solid #1B3A6B;
        padding: 16px 20px; margin: 24px 0;
        font-size: 13px; line-height: 1.6; color: #0A0A0A;
    }
    .finding-label {
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.15em;
        color: #1B3A6B; margin-bottom: 6px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 0 48px 48px 48px;
        max-width: 1400px;
    }
</style>
<div class="top-bar"></div>
""", unsafe_allow_html=True)

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

# ── BASE PLOT LAYOUT (no axis keys) ──────────────────────────
PLOT_BASE = dict(
    font=dict(family="IBM Plex Sans", size=12, color="#0A0A0A"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=0, r=0, t=24, b=0),
    legend=dict(
        orientation="h", yanchor="bottom",
        y=1.02, xanchor="left", x=0,
        font=dict(size=11)
    )
)

AXIS_STYLE = dict(
    gridcolor="#F0F0F0",
    linecolor="#E5E5E5",
    tickfont=dict(family="IBM Plex Mono", size=11)
)

# ── HEADER ────────────────────────────────────────────────────
col_hdr, col_aff = st.columns([3, 1])
with col_hdr:
    st.markdown("""
    <div class="ares-header">
        <div class="ares-wordmark">▲ ARES</div>
        <div class="ares-title">Adaptive Relationship &amp; Event Simulator</div>
        <div class="ares-sub">
            US–Iran–Israel Conflict Network &nbsp;·&nbsp;
            Oct 2023 → Aug 2026 &nbsp;·&nbsp;
            Structural Balance Theory (Marvel–Kleinberg–Strogatz 2011)
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_aff:
    st.markdown("""
    <div class="ares-header" style="text-align:right">
        <div class="ares-affil">
            IIM CALCUTTA<br>SNA · GROUP 5<br>MBA BATCH 62
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── NAVIGATION TABS ───────────────────────────────────────────
tabs = st.tabs([
    "OVERVIEW",
    "NETWORK",
    "TRAJECTORIES",
    "MEDIATOR ANALYSIS",
    "EVENT LOG"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tabs[0]:

    hostile_count = int((EVENTS_DF["goldstein"] < 0).sum())
    diplo_count   = int((EVENTS_DF["goldstein"] > 0).sum())
    iran_start    = float(SIM_LOG.iloc[0]["iran_might"])
    iran_end      = float(SIM_LOG.iloc[-1]["iran_might"])
    iran_loss     = (
        (1 - iran_end / iran_start) * 100
        if iran_start > 0 else 0
    )
    final_med = SIM_LOG.iloc[-1]["top_mediator"]

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Total Events</div>
            <div class="stat-value">{len(EVENTS_DF)}</div>
            <div class="stat-delta">Oct 2023 – Aug 2026</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Hostile Events</div>
            <div class="stat-value threat">{hostile_count}</div>
            <div class="stat-delta">{hostile_count/len(EVENTS_DF)*100:.0f}% of all events</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Diplomatic Events</div>
            <div class="stat-value stable">{diplo_count}</div>
            <div class="stat-delta">{diplo_count/len(EVENTS_DF)*100:.0f}% of all events</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Iran Might Loss</div>
            <div class="stat-value threat">{iran_loss:.0f}%</div>
            <div class="stat-delta">{iran_start:.3f} → {iran_end:.3f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Structural Mediator</div>
            <div class="stat-value" style="font-size:20px">{final_med}</div>
            <div class="stat-delta">by network position</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # phases
    st.markdown(
        '<div class="section-header">Conflict Phases</div>',
        unsafe_allow_html=True
    )
    phases = [
        ("PHASE I",   "Oct 2023 – Dec 2023",
         "Hamas Oct 7 attack triggers regional cascade. "
         "Houthis and Hezbollah activate across multiple "
         "fronts. US expands Iran sanctions."),
        ("PHASE II",  "Jan 2024 – Oct 2024",
         "First direct Iran–Israel military exchanges. "
         "Iran launches 300+ drones and missiles. Israel "
         "retaliates. Nasrallah and Sinwar killed."),
        ("PHASE III", "Nov 2024 – Feb 2026",
         "Lebanon ceasefire. Assad regime falls — Iran "
         "loses Syria. Twelve-Day War (Jun 2025): US joins "
         "Israel strikes on Iranian nuclear sites."),
        ("PHASE IV",  "Feb 2026 – Present",
         "Operation Epic Fury: 900 strikes, Khamenei "
         "killed. Strait of Hormuz crisis. Pakistan brokers "
         "ceasefire. 14-point MOU signed."),
    ]
    for pid, dates, desc in phases:
        st.markdown(f"""
        <div class="phase-row">
            <div class="phase-id">{pid}</div>
            <div class="phase-date">{dates}</div>
            <div class="phase-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # might trajectory
    st.markdown(
        '<div class="section-header">National Power Trajectory</div>',
        unsafe_allow_html=True
    )

    fig_might = go.Figure()
    for country, color, dash in [
        ("Iran",   "#C41E3A", "solid"),
        ("Israel", "#1B3A6B", "solid"),
        ("USA",    "#1B7A3E", "dash"),
    ]:
        col_name = f"{country.lower()}_might"
        if col_name in SIM_LOG.columns:
            fig_might.add_trace(go.Scatter(
                x=SIM_LOG["date"],
                y=SIM_LOG[col_name],
                name=country,
                line=dict(color=color, width=2, dash=dash),
                mode="lines"
            ))

    for d, lbl in [
        ("2024-04-13", "Direct exchange"),
        ("2025-06-13", "12-Day War"),
        ("2026-02-28", "Epic Fury"),
        ("2026-04-08", "Ceasefire"),
    ]:
        fig_might.add_vline(
            x=d, line_dash="dot",
            line_color="#CCCCCC", line_width=1,
            annotation_text=lbl,
            annotation_textangle=-90,
            annotation_font=dict(size=9, color="#999999")
        )

    fig_might.update_layout(**PLOT_BASE, height=260)
    fig_might.update_xaxes(**AXIS_STYLE, title_text="")
    fig_might.update_yaxes(
        **AXIS_STYLE,
        title_text="Might Score [0 – 1]",
        range=[0, 1.05]
    )
    st.plotly_chart(fig_might, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — NETWORK
# ═══════════════════════════════════════════════════════════════
with tabs[1]:

    col_ctrl, col_main = st.columns([1, 4])
    with col_ctrl:
        st.markdown(
            '<div class="section-header">Controls</div>',
            unsafe_allow_html=True
        )
        net_choice = st.radio(
            "Network state",
            ["Initial — Oct 2023", "Final — Aug 2026"],
            label_visibility="collapsed"
        )
        show_thresh = st.slider(
            "Edge threshold", 0.0, 1.0, 0.25, 0.05,
            help="Hide edges weaker than this value"
        )
        st.markdown("---")
        st.markdown("""
        <div style="font-size:12px;color:#6B6B6B;line-height:2">
        <span style="color:#1B3A6B;font-size:16px">●</span>
        US-aligned<br>
        <span style="color:#C41E3A;font-size:16px">●</span>
        Iran-aligned<br>
        <span style="color:#888888;font-size:16px">●</span>
        Straddler<br><br>
        <span style="color:#1B7A3E">—</span> Friendly<br>
        <span style="color:#C41E3A">—</span> Hostile<br><br>
        Node size = Might
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        G_show = (
            G_INITIAL if "Initial" in net_choice
            else G_FINAL
        )

        nodes_sorted = sorted(
            G_show.nodes(data=True),
            key=lambda x: x[1].get("alignment", 0),
            reverse=True
        )
        n_nodes = len(nodes_sorted)
        pos = {}
        for idx, (node, _) in enumerate(nodes_sorted):
            angle = 2 * np.pi * idx / n_nodes
            pos[node] = (np.cos(angle), np.sin(angle))

        edge_traces = []
        for a, b, edata in G_show.edges(data=True):
            rel = edata.get("relationship", 0)
            if abs(rel) < show_thresh:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            opacity = min(0.9, 0.2 + abs(rel) * 0.7)
            color = (
                f"rgba(27,122,62,{opacity})"
                if rel >= 0 else
                f"rgba(196,30,58,{opacity})"
            )
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=max(0.5, abs(rel) * 2.5),
                    color=color
                ),
                hoverinfo="none", showlegend=False
            ))

        node_x, node_y = [], []
        node_text, node_hover = [], []
        node_color, node_size = [], []

        for node, ndata in G_show.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            al = ndata.get("alignment", 0)
            m  = ndata.get("might", 0.1)
            camp = (
                "US" if al > 0.15 else
                ("Iran" if al < -0.15 else "Neutral")
            )
            node_hover.append(
                f"<b>{node}</b><br>"
                f"Might: {m:.3f}<br>"
                f"Alignment: {al:+.3f}<br>"
                f"Camp: {camp}"
            )
            if al > 0.15:
                node_color.append("#1B3A6B")
            elif al < -0.15:
                node_color.append("#C41E3A")
            else:
                node_color.append("#888888")
            node_size.append(12 + 35 * float(m))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(
                family="IBM Plex Sans",
                size=10, color="#0A0A0A"
            ),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=1.5, color="#FFFFFF")
            ),
            showlegend=False
        )

        label = (
            "Initial Network — Oct 2023"
            if "Initial" in net_choice
            else "Final Network — Aug 2026"
        )
        fig_net = go.Figure(
            data=edge_traces + [node_trace]
        )
        fig_net.update_layout(
            height=580,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=0, r=0, t=32, b=0),
            font=dict(family="IBM Plex Sans"),
            title=dict(
                text=label,
                font=dict(
                    size=13, color="#6B6B6B",
                    family="IBM Plex Sans"
                ),
                x=0
            )
        )
        fig_net.update_xaxes(
            showgrid=False, zeroline=False,
            showticklabels=False, scaleanchor="y"
        )
        fig_net.update_yaxes(
            showgrid=False, zeroline=False,
            showticklabels=False
        )
        st.plotly_chart(fig_net, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — TRAJECTORIES
# ═══════════════════════════════════════════════════════════════
with tabs[2]:

    st.markdown(
        '<div class="section-header">Network Tension Over Time</div>',
        unsafe_allow_html=True
    )

    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=SIM_LOG["date"],
        y=SIM_LOG["balance_energy"],
        name="Balance Energy",
        line=dict(color="#1B3A6B", width=2),
        mode="lines"
    ))
    fig_traj.add_trace(go.Scatter(
        x=SIM_LOG["date"],
        y=SIM_LOG["frac_unbalanced"],
        name="Fraction Unbalanced",
        line=dict(color="#C41E3A", width=2),
        mode="lines"
    ))
    fig_traj.add_trace(go.Scatter(
        x=SIM_LOG["date"],
        y=SIM_LOG["top_score"],
        name="Top Mediator Score",
        line=dict(color="#888888", width=1.5, dash="dot"),
        mode="lines"
    ))

    for d, lbl in [
        ("2024-04-13", "First direct exchange"),
        ("2025-06-13", "Twelve-Day War"),
        ("2026-02-28", "Operation Epic Fury"),
        ("2026-04-08", "Pakistan ceasefire"),
    ]:
        fig_traj.add_vline(
            x=d, line_dash="dot",
            line_color="#CCCCCC", line_width=1,
            annotation_text=lbl,
            annotation_textangle=-90,
            annotation_font=dict(size=9, color="#999")
        )

    fig_traj.update_layout(**PLOT_BASE, height=340)
    fig_traj.update_xaxes(**AXIS_STYLE, title_text="")
    fig_traj.update_yaxes(**AXIS_STYLE, title_text="Value")
    st.plotly_chart(fig_traj, use_container_width=True)

    st.markdown(
        '<div class="section-header">Mediator Leadership — Event Count</div>',
        unsafe_allow_html=True
    )

    med_counts = (
        SIM_LOG["top_mediator"]
        .value_counts()
        .reset_index()
    )
    med_counts.columns = ["country", "count"]
    med_counts = med_counts.sort_values(
        "count", ascending=True
    )

    top_country = med_counts.iloc[-1]["country"]
    fig_bar = go.Figure(go.Bar(
        x=med_counts["count"],
        y=med_counts["country"],
        orientation="h",
        marker=dict(
            color=[
                "#C41E3A" if c == top_country
                else "#E5E5E5"
                for c in med_counts["country"]
            ],
            line=dict(width=0)
        ),
        text=med_counts["count"],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11)
    ))
    fig_bar.update_layout(
        **PLOT_BASE, height=340, showlegend=False
    )
    fig_bar.update_xaxes(
        **AXIS_STYLE,
        title_text="Events as top mediator"
    )
    fig_bar.update_yaxes(**AXIS_STYLE, title_text="")
    st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — MEDIATOR ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tabs[3]:

    col_snap, _ = st.columns([2, 1])
    with col_snap:
        tick_idx = st.slider(
            "Scroll through the conflict timeline",
            min_value=0,
            max_value=len(SIM_LOG) - 1,
            value=len(SIM_LOG) - 1
        )

    selected  = SIM_LOG.iloc[tick_idx]
    top3_raw  = selected["top3"]
    snap_date = selected["date"]
    snap_evt  = selected["event"]

    st.markdown(
        f'<div class="section-header">'
        f'Mediator Ranking · {snap_date}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-size:12px;color:#6B6B6B;'
        f'margin-bottom:20px">{snap_evt}</div>',
        unsafe_allow_html=True
    )

    if len(top3_raw) >= 3:
        cards = ""
        for i, (name, score) in enumerate(top3_raw[:3]):
            top_cls = "top" if i == 0 else ""
            rank_lbl = ["RANK 01", "RANK 02", "RANK 03"][i]
            cards += f"""
            <div class="podium-card">
                <div class="podium-rank">{rank_lbl}</div>
                <div class="podium-country {top_cls}">
                    {name}
                </div>
                <div class="podium-score">{score:.4f}</div>
            </div>
            """
        st.markdown(
            f'<div class="podium-grid">{cards}</div>',
            unsafe_allow_html=True
        )

    # key moments table
    st.markdown(
        '<div class="section-header">Mediator at Key Moments</div>',
        unsafe_allow_html=True
    )

    key_moments = [
        ("2023-10-07", "Oct 7 — Hamas attack"),
        ("2024-04-13", "First direct Iran–Israel exchange"),
        ("2024-10-01", "Iran fires 180 ballistic missiles"),
        ("2025-06-13", "Twelve-Day War begins"),
        ("2026-02-28", "Operation Epic Fury"),
        ("2026-04-08", "Pakistan brokers ceasefire"),
    ]

    rows = ""
    for d, label in key_moments:
        match = SIM_LOG[SIM_LOG["date"] >= d]
        if not match.empty:
            row   = match.iloc[0]
            med   = row["top_mediator"]
            score = row["top_score"]
            rows += f"""
            <div class="event-row">
                <div class="event-date">{d}</div>
                <div class="event-actor">{med}</div>
                <div class="event-score"
                     style="color:#1B3A6B">{score:.4f}</div>
                <div class="event-desc"
                     style="grid-column:4/6">{label}</div>
            </div>
            """
    st.markdown(rows, unsafe_allow_html=True)

    # finding
    st.markdown("""
    <div class="finding-box">
        <div class="finding-label">Key Finding</div>
        Our structural model identifies <strong>India and China</strong>
        as best-positioned mediators by network topology — large states
        maintaining non-hostile ties to both camps throughout the
        conflict. Real-world mediation was carried out by
        <strong>Qatar</strong> (2023 hostage deal, 2025 Gaza ceasefire),
        <strong>Oman</strong> (US–Iran back-channel, Strait of Hormuz
        MOU), and <strong>Pakistan</strong> (Apr 2026 ceasefire,
        Jun 2026 14-point MOU). This divergence is itself a finding:
        structural position is <em>necessary but not sufficient</em>
        for mediation. Political trust, geographic leverage, and
        pre-existing relationships are not captured by the network
        model — pointing to clear extensions.
    </div>
    """, unsafe_allow_html=True)

    col_model, col_real = st.columns(2)

    with col_model:
        st.markdown(
            '<div class="section-header">Model Output</div>',
            unsafe_allow_html=True
        )
        final_med_data = SIM_LOG.iloc[-1]["top3"]
        for rank, (name, score) in enumerate(
                final_med_data, 1):
            bar_w = min(int(score * 800), 280)
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;
                            justify-content:space-between;
                            font-size:12px;margin-bottom:4px">
                    <span><b>#{rank} {name}</b></span>
                    <span style="font-family:'IBM Plex Mono';
                                 color:#6B6B6B">{score:.4f}</span>
                </div>
                <div style="background:#F0F0F0;height:3px;
                            width:100%">
                    <div style="background:#1B3A6B;
                                height:3px;width:{bar_w}px">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_real:
        st.markdown(
            '<div class="section-header">Real-World Mediators</div>',
            unsafe_allow_html=True
        )
        real = [
            ("Qatar",
             "Nov 2023 hostage deal · Jan 2025 Gaza ceasefire"),
            ("Oman",
             "2025 US–Iran back-channel · Strait of Hormuz MOU"),
            ("Pakistan",
             "Apr 2026 ceasefire · Jun 2026 14-point MOU"),
        ]
        for name, roles in real:
            st.markdown(f"""
            <div style="margin-bottom:16px">
                <div style="font-size:14px;font-weight:600;
                            color:#0A0A0A;margin-bottom:2px">
                    {name}
                </div>
                <div style="font-size:12px;color:#6B6B6B;
                            line-height:1.5">{roles}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — EVENT LOG
# ═══════════════════════════════════════════════════════════════
with tabs[4]:

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        show_hostile = st.checkbox("Hostile events", True)
    with col_f2:
        show_diplo   = st.checkbox("Diplomatic events", True)
    with col_f3:
        search_actor = st.text_input(
            "Filter by actor", placeholder="e.g. Iran"
        )

    filtered = EVENTS_DF.copy()
    if not show_hostile:
        filtered = filtered[filtered["goldstein"] >= 0]
    if not show_diplo:
        filtered = filtered[filtered["goldstein"] <= 0]
    if search_actor:
        mask = (
            filtered["actor1"].str.contains(
                search_actor, case=False, na=False
            ) |
            filtered["actor2"].str.contains(
                search_actor, case=False, na=False
            )
        )
        filtered = filtered[mask]

    st.markdown(
        f'<div class="section-header">'
        f'{len(filtered)} Events</div>',
        unsafe_allow_html=True
    )

    fig_gold = go.Figure()
    fig_gold.add_bar(
        x=filtered["date"].astype(str),
        y=filtered["goldstein"],
        marker_color=[
            "#C41E3A" if g < 0 else "#1B7A3E"
            for g in filtered["goldstein"]
        ],
        hovertext=(
            filtered["actor1"] + " → " +
            filtered["actor2"] + "<br>" +
            filtered["description"]
        ),
        hoverinfo="text+y"
    )
    fig_gold.update_layout(
        **PLOT_BASE, height=220,
        showlegend=False, bargap=0.15
    )
    fig_gold.update_xaxes(**AXIS_STYLE, title_text="")
    fig_gold.update_yaxes(
        **AXIS_STYLE, title_text="Goldstein Scale"
    )
    st.plotly_chart(fig_gold, use_container_width=True)

    header_row = """
    <div class="event-row" style="font-size:10px;
         font-weight:600;text-transform:uppercase;
         letter-spacing:0.1em;color:#6B6B6B;
         border-bottom:2px solid #E5E5E5">
        <div>Date</div><div>Actor 1</div>
        <div>Actor 2</div><div>Description</div>
        <div>Score</div>
    </div>
    """
    event_rows = ""
    for _, row in filtered.iterrows():
        g    = row["goldstein"]
        cls  = "hostile" if g < 0 else "diplo"
        sign = "+" if g >= 0 else ""
        event_rows += f"""
        <div class="event-row">
            <div class="event-date">
                {str(row['date'].date())}
            </div>
            <div class="event-actor">{row['actor1']}</div>
            <div class="event-actor">{row['actor2']}</div>
            <div class="event-desc">{row['description']}</div>
            <div class="event-score {cls}">
                {sign}{g:.1f}
            </div>
        </div>
        """
    st.markdown(header_row + event_rows, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:64px;padding-top:20px;
            border-top:1px solid #E5E5E5;
            display:flex;justify-content:space-between;
            font-size:11px;color:#9B9B9B;
            font-family:'IBM Plex Mono',monospace">
    <span>ARES · Adaptive Relationship &amp; Event Simulator</span>
    <span>Data: World Bank · Guardian API · GDELT · UN Voting Records</span>
    <span>IIM Calcutta · SNA Group 5 · MBA Batch 62 · 2026</span>
</div>
""", unsafe_allow_html=True)
