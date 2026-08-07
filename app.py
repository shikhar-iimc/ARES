
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pickle
import os

st.set_page_config(
    page_title="ARES — Adaptive Relationship & Event Simulator",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ── STYLES ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .stMetric { background: #1a1a2e; border-radius: 8px;
                padding: 10px; border: 1px solid #16213e; }
    .title-text { font-size: 2.5rem; font-weight: 800;
                  color: #e94560; letter-spacing: 2px; }
    .subtitle-text { font-size: 1rem; color: #aaaaaa; }
    .phase-box { background: #16213e; border-radius: 6px;
                 padding: 8px 12px; margin: 4px 0;
                 border-left: 3px solid #e94560; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.markdown(
        "<div class='title-text'>⚔️ ARES</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle-text'>"
        "Adaptive Relationship & Event Simulator | "
        "US–Iran–Israel Conflict Network | "
        "Oct 2023 → Aug 2026"
        "</div>",
        unsafe_allow_html=True
    )
with col_logo:
    st.markdown("###")
    st.markdown("**IIM Calcutta** | SNA Group 5")

st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    view_mode = st.radio(
        "View Mode",
        ["📊 Overview", "🌐 Network", "📈 Trajectories",
         "🏆 Mediator Rankings", "📅 Event Timeline"],
        index=0
    )

    st.divider()
    st.markdown("### 🔍 Select Time Point")
    tick_idx = st.slider(
        "Event index",
        min_value=0,
        max_value=len(SIM_LOG)-1,
        value=len(SIM_LOG)-1,
        help="Scroll through the conflict timeline"
    )
    selected = SIM_LOG.iloc[tick_idx]

    st.divider()
    st.markdown("### 📌 Current Snapshot")
    st.markdown(f"**Date:** {selected['date']}")
    st.markdown(f"**Event:** {selected['event'][:50]}...")
    st.markdown(
        f"**Top Mediator:** {selected['top_mediator']}"
    )
    st.markdown(
        f"**Energy:** {selected['balance_energy']:.4f}"
    )

# ── OVERVIEW ──────────────────────────────────────────────────
if view_mode == "📊 Overview":
    st.subheader("Conflict at a Glance")

    # key metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.metric("Total Events", len(EVENTS_DF))
    with c2:
        hostile = (EVENTS_DF["goldstein"] < 0).sum()
        st.metric("Hostile Events", hostile)
    with c3:
        diplo = (EVENTS_DF["goldstein"] > 0).sum()
        st.metric("Diplomatic Events", diplo)
    with c4:
        iran_might_drop = (
            1 - SIM_LOG.iloc[-1]["iran_might"] /
            SIM_LOG.iloc[0]["iran_might"]
        ) * 100
        st.metric(
            "Iran Might Loss",
            f"{iran_might_drop:.0f}%",
            delta=f"-{iran_might_drop:.0f}%"
        )
    with c5:
        final_mediator = SIM_LOG.iloc[-1]["top_mediator"]
        st.metric("Current Best Mediator", final_mediator)

    st.divider()

    # conflict phases
    st.subheader("Four Phases of Conflict")
    phases = [
        ("Phase 1", "Oct 2023 – Dec 2023",
         "Hamas Oct 7 attack triggers regional cascade. "
         "Houthis, Hezbollah activate. US sanctions Iran.",
         "#e94560"),
        ("Phase 2", "Jan 2024 – Oct 2024",
         "First direct Iran-Israel military exchanges. "
         "300 drones/missiles. Israel retaliates. "
         "Nasrallah killed.",
         "#f5a623"),
        ("Phase 3", "Nov 2024 – Feb 2026",
         "Lebanon ceasefire. Assad falls. Twelve-Day War "
         "Jun 2025. US joins Israel strikes on Iran nuclear "
         "sites.",
         "#7ed321"),
        ("Phase 4", "Feb 2026 – Present",
         "Operation Epic Fury. 900 strikes. Khamenei killed. "
         "Pakistan brokers ceasefire. Strait of Hormuz "
         "crisis. MOU signed.",
         "#4a90e2"),
    ]

    cols = st.columns(4)
    for col, (phase, dates, desc, color) in zip(
            cols, phases):
        with col:
            st.markdown(
                f"<div style='background:#1a1a2e;"
                f"border-radius:8px;padding:12px;"
                f"border-top:3px solid {color}'>"
                f"<b style='color:{color}'>{phase}</b><br>"
                f"<small style='color:#aaa'>{dates}</small>"
                f"<p style='font-size:0.8rem;margin-top:8px'>"
                f"{desc}</p></div>",
                unsafe_allow_html=True
            )

    st.divider()

    # might degradation
    st.subheader("National Power (Might) Over Conflict")
    fig_might = go.Figure()
    for country, color in [
        ("Iran",   "#e94560"),
        ("Israel", "#4a90e2"),
        ("USA",    "#7ed321"),
    ]:
        col_name = f"{country.lower()}_might"
        if col_name in SIM_LOG.columns:
            fig_might.add_trace(go.Scatter(
                x=SIM_LOG["date"],
                y=SIM_LOG[col_name],
                name=country,
                line=dict(color=color, width=2),
                mode="lines"
            ))
    fig_might.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_title="Date",
        yaxis_title="Might Score [0,1]",
        legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig_might, use_container_width=True)

# ── NETWORK VIEW ──────────────────────────────────────────────
elif view_mode == "🌐 Network":
    st.subheader("Conflict Network Visualization")

    net_choice = st.radio(
        "Show network at:",
        ["Initial (Oct 2023)", "Final (Aug 2026)"],
        horizontal=True
    )
    G_show = G_INITIAL if "Initial" in net_choice              else G_FINAL

    show_thresh = st.slider(
        "Edge threshold (hide weak ties)",
        0.0, 1.0, 0.2, 0.05
    )

    # build plotly network
    pos = nx.circular_layout(
        G_show,
        scale=1
    )

    # sort by alignment for arc layout
    nodes_sorted = sorted(
        G_show.nodes(data=True),
        key=lambda x: x[1].get("alignment", 0),
        reverse=True
    )
    n_nodes = len(nodes_sorted)
    for idx, (node, _) in enumerate(nodes_sorted):
        angle = 2 * np.pi * idx / n_nodes
        pos[node] = (np.cos(angle), np.sin(angle))

    # edges
    edge_traces = []
    for a, b, edata in G_show.edges(data=True):
        rel = edata.get("relationship", 0)
        if abs(rel) < show_thresh:
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        color = (
            f"rgba(0,200,100,{min(1,abs(rel))})"
            if rel >= 0 else
            f"rgba(220,50,50,{min(1,abs(rel))})"
        )
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=abs(rel)*3, color=color),
            hoverinfo="none",
            showlegend=False
        ))

    # nodes
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
        node_hover.append(
            f"{node}<br>"
            f"Might: {m:.3f}<br>"
            f"Alignment: {al:+.3f}"
        )
        if al > 0.15:
            node_color.append("#4a90e2")
        elif al < -0.15:
            node_color.append("#e94560")
        else:
            node_color.append("#aaaaaa")
        node_size.append(10 + 30 * m)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color="white")
        ),
        showlegend=False
    )

    fig_net = go.Figure(
        data=edge_traces + [node_trace]
    )
    fig_net.update_layout(
        template="plotly_dark",
        height=600,
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False),
        margin=dict(l=0,r=0,t=20,b=0),
        annotations=[
            dict(text="🔵 US Camp  🔴 Iran Camp  ⚪ Straddler",
                 x=0.5, y=-0.05, xref="paper",
                 yref="paper", showarrow=False,
                 font=dict(color="white", size=12))
        ]
    )
    st.plotly_chart(fig_net, use_container_width=True)

# ── TRAJECTORIES ──────────────────────────────────────────────
elif view_mode == "📈 Trajectories":
    st.subheader("Network Tension Over Time")

    fig_energy = go.Figure()

    # balance energy
    fig_energy.add_trace(go.Scatter(
        x=SIM_LOG["date"],
        y=SIM_LOG["balance_energy"],
        name="Balance Energy",
        line=dict(color="#7ed321", width=2),
        mode="lines"
    ))

    # frac unbalanced
    fig_energy.add_trace(go.Scatter(
        x=SIM_LOG["date"],
        y=SIM_LOG["frac_unbalanced"],
        name="Frac Unbalanced",
        line=dict(color="#f5a623", width=2),
        mode="lines"
    ))

    # top mediator score
    fig_energy.add_trace(go.Scatter(
        x=SIM_LOG["date"],
        y=SIM_LOG["top_score"],
        name="Top Mediator Score",
        line=dict(color="#e94560", width=2,
                  dash="dot"),
        mode="lines"
    ))

    # add phase markers
    phase_dates = [
        ("2024-04-13", "First direct exchange"),
        ("2025-06-13", "12-Day War"),
        ("2026-02-28", "Operation Epic Fury"),
        ("2026-04-08", "Pakistan ceasefire"),
    ]
    for d, label in phase_dates:
        fig_energy.add_vline(
            x=d,
            line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text=label,
            annotation_textangle=-90,
            annotation_font_size=9
        )

    fig_energy.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h"),
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig_energy, use_container_width=True)

    st.divider()
    st.subheader("Mediator Evolution")

    # who was top mediator at each event
    mediator_counts = (
        SIM_LOG["top_mediator"]
        .value_counts()
        .reset_index()
    )
    mediator_counts.columns = ["country", "events_as_top"]

    fig_med = px.bar(
        mediator_counts,
        x="country",
        y="events_as_top",
        color="events_as_top",
        color_continuous_scale="Reds",
        template="plotly_dark",
        labels={"events_as_top": "Events as Top Mediator"}
    )
    fig_med.update_layout(
        height=300,
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig_med, use_container_width=True)

# ── MEDIATOR RANKINGS ─────────────────────────────────────────
elif view_mode == "🏆 Mediator Rankings":
    st.subheader("Mediator Leaderboard")

    st.info(
        "**Mediator Score** = balance (non-hostile to both "
        "camps) × bridge (network reach) × credibility "
        "(might). Countries hostile to either camp score 0.",
        icon="ℹ️"
    )

    # current snapshot rankings from sim log
    snapshot   = SIM_LOG.iloc[tick_idx]
    top3_raw   = snapshot["top3"]

    st.markdown(
        f"### Rankings at: **{snapshot['date']}**"
    )
    st.markdown(f"*Event: {snapshot['event']}*")
    st.divider()

    # display top 3 as podium
    if len(top3_raw) >= 3:
        p1, p2, p3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for col, medal, (name, score) in zip(
                [p2, p1, p3],
                ["🥈","🥇","🥉"],
                [top3_raw[1], top3_raw[0], top3_raw[2]]):
            with col:
                st.markdown(
                    f"<div style='text-align:center;"
                    f"background:#1a1a2e;border-radius:12px;"
                    f"padding:20px'>"
                    f"<div style='font-size:2rem'>{medal}</div>"
                    f"<div style='font-size:1.4rem;"
                    f"font-weight:800;color:#e94560'>"
                    f"{name}</div>"
                    f"<div style='color:#aaa'>score: "
                    f"{score:.4f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # real world comparison
    st.subheader("Model vs Reality")
    col_model, col_real = st.columns(2)

    with col_model:
        st.markdown("**ARES Model (structural position)**")
        for rank, (name, score) in enumerate(
                top3_raw, 1):
            st.markdown(
                f"#{rank} **{name}** — {score:.4f}"
            )

    with col_real:
        st.markdown("**Real-world mediators**")
        real_mediators = [
            ("Qatar", "Nov 2023 hostage deal",
             "Jan 2025 Gaza ceasefire"),
            ("Oman",  "2025 US-Iran back-channel",
             "2026 Strait of Hormuz MOU"),
            ("Pakistan", "Apr 2026 ceasefire",
             "Jun 2026 14-point MOU"),
        ]
        for name, role1, role2 in real_mediators:
            st.markdown(
                f"**{name}** — {role1}; {role2}"
            )

    st.divider()
    st.markdown(
        "**Finding:** Structural position (India, China) "
        "differs from real-world mediation outcomes "
        "(Qatar, Oman, Pakistan). "
        "This divergence shows that network position is "
        "*necessary but not sufficient* — political trust, "
        "geography, and pre-existing relationships also "
        "matter. This points to future model extensions."
    )

# ── EVENT TIMELINE ────────────────────────────────────────────
elif view_mode == "📅 Event Timeline":
    st.subheader("Complete Conflict Timeline")

    # filter options
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        show_hostile = st.checkbox("Show hostile events", True)
    with col_f2:
        show_diplo   = st.checkbox("Show diplomatic events",
                                    True)

    filtered = EVENTS_DF.copy()
    if not show_hostile:
        filtered = filtered[filtered["goldstein"] >= 0]
    if not show_diplo:
        filtered = filtered[filtered["goldstein"] <= 0]

    # goldstein chart
    fig_gold = go.Figure()
    fig_gold.add_bar(
        x=filtered["date"].astype(str),
        y=filtered["goldstein"],
        marker_color=[
            "#e94560" if g < 0 else "#7ed321"
            for g in filtered["goldstein"]
        ],
        text=filtered["actor1"] + "→" + filtered["actor2"],
        hovertext=filtered["description"],
        hoverinfo="text+y"
    )
    fig_gold.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_title="Date",
        yaxis_title="Goldstein Scale",
        margin=dict(l=0,r=0,t=20,b=0)
    )
    st.plotly_chart(fig_gold, use_container_width=True)

    # event table
    st.dataframe(
        filtered[[
            "date", "actor1", "actor2",
            "event_type", "goldstein", "description"
        ]].rename(columns={
            "date": "Date",
            "actor1": "Actor 1",
            "actor2": "Actor 2",
            "event_type": "Type",
            "goldstein": "Goldstein",
            "description": "Description"
        }),
        use_container_width=True,
        height=400
    )

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#555;"
    "font-size:0.8rem'>"
    "ARES — Adaptive Relationship & Event Simulator | "
    "IIM Calcutta MBA Batch 62 | SNA Group 5 | "
    "Data: World Bank API + Guardian API + GDELT | "
    "Model: Structural Balance Theory "
    "(Marvel-Kleinberg-Strogatz 2011)"
    "</div>",
    unsafe_allow_html=True
)
