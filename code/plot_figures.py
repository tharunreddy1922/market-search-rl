"""
============================================================
  plot_figures.py — Figure Generator
  Market Search and Purchase Scheduling Using RL

  P. Tharun Reddy | A00051705 | MSc Data Science
  University of Roehampton | 2026
============================================================

Generates all 8 dissertation figures as PNG files.

HOW TO USE:
  # Option 1 — Run after main.py (reads results.json automatically):
      python main.py          # trains + evaluates all agents
      python plot_figures.py  # generates figures from results

  # Option 2 — Called automatically at end of main.py (default behaviour)

  # Option 3 — Standalone with custom data:
      python plot_figures.py --data path/to/results.json

OUTPUT:
  figures/fig1_environment.png
  figures/fig2_reward_comparison.png
  figures/fig3_radar.png
  figures/fig4_experiments.png
  figures/fig5_scatter.png
  figures/fig6_price_quality.png
  figures/fig7_dqn_architecture.png
  figures/fig8_convergence.png

MODIFYING FIGURES:
  Each function below has a clearly marked DATA section.
  Edit only the values inside that section to change labels,
  add/remove agents, or update numbers.
"""

import os
import sys
import json
import argparse
import numpy as np

# ── Matplotlib setup (headless — no display needed) ──────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Output directory ──────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'figures')

# ── Shared style ──────────────────────────────────────────────────────────────
BG       = "#f4f6f9"
GRID_CLR = "#cccccc"
TITLE_FS = 13
LABEL_FS = 10

# Category colours (used across multiple figures)
CAT_COLORS = {
    "Deep RL":    "#1f77b4",
    "Tabular RL": "#d95f02",
    "Heuristic":  "#2ca02c",
    "PPO":        "#e6ac00",
    "Random":     "#888888",
}


def _save(fig, name):
    """Save figure to the figures/ folder and close it."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  ✓  {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — MDP Environment: Star-Topology Store Network
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig1():
    # ── DATA ───────────────────────────────────────────────────────────────────
    stores = [
        ("Store 1", ( 0.85,  1.10)),
        ("Store 2", (-0.85,  1.10)),
        ("Store 3", (-1.25,  0.00)),
        ("Store 4", (-0.85, -1.10)),
        ("Store 5", ( 0.85, -1.10)),
        ("Store 0", ( 1.25,  0.00)),
    ]
    store_detail = "3 brands/good\n60% avail."
    travel_time  = "5 min"
    bullets = [
        "20 goods per list",
        "5 brands per good",
        "120-min budget",
        "±30% price premium",
        "7-day expiry window",
    ]
    # ──────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.65, 1.85)
    ax.set_ylim(-1.65, 1.65)
    ax.axis('off')

    hub_color   = "#1a3a5c"
    store_color = "#4daadf"

    for name, (x, y) in stores:
        # Arrow hub → store
        ax.annotate("",
                    xy=(x * 0.72, y * 0.72),
                    xytext=(x * 0.55, y * 0.55),
                    arrowprops=dict(arrowstyle="-|>", color="#4daadf",
                                   lw=1.6, mutation_scale=16))
        # Travel time label
        mid_x = (x * 0.55 + x * 0.72) / 2 + (-0.12 if x < 0 else 0.06)
        mid_y = (y * 0.55 + y * 0.72) / 2
        ax.text(mid_x, mid_y, travel_time, fontsize=8.5,
                color="#2980b9", ha='center', va='center')
        # Store circle
        ax.add_patch(plt.Circle((x, y), 0.27, color=store_color, zorder=3))
        ax.text(x, y + 0.05, name, fontsize=10, fontweight='bold',
                color='white', ha='center', va='center', zorder=4)
        ax.text(x, y - 0.09, store_detail, fontsize=7.5,
                color='white', ha='center', va='center', zorder=4)

    # Hub
    ax.add_patch(plt.Circle((0, 0), 0.22, color=hub_color, zorder=5))
    ax.text(0,  0.04, "Hub",   fontsize=11, fontweight='bold',
            color='white',    ha='center', va='center', zorder=6)
    ax.text(0, -0.07, "Agent", fontsize=9,
            color='#aac8e8',  ha='center', va='center', zorder=6)

    # Bullet list
    bullet_text = "\n\n".join(f"• {b}" for b in bullets)
    ax.text(1.48, 0.15, bullet_text, fontsize=10, color='#333',
            ha='left', va='center', linespacing=1.8)

    ax.set_title("MDP Environment — Star-Topology Store Network",
                 fontsize=TITLE_FS, fontweight='bold', pad=14)
    plt.tight_layout()
    _save(fig, "fig1_environment.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Mean Cumulative Reward — All Agents (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig2(results=None):
    # ── DATA (overridden by live results if passed in) ─────────────────────────
    # Format: (display_name, reward_value, category)
    agents_static = [
        ("Dueling-DQN",    498.66, "Deep RL"),
        ("Double-DQN",     475.40, "Deep RL"),
        ("DQN",            472.93, "Deep RL"),
        ("SARSA",          375.77, "Tabular RL"),
        ("Q-Learning",     355.00, "Tabular RL"),
        ("Best-Deal",      338.47, "Heuristic"),
        ("Greedy",         301.64, "Heuristic"),
        ("TimeAware",      254.37, "Heuristic"),
        ("PPO",            184.17, "PPO"),
        ("Freshness-1st",  152.59, "Heuristic"),
        ("Prof-Heuristic", 148.69, "Heuristic"),
        ("Random",         -50.88, "Random"),
    ]
    # ──────────────────────────────────────────────────────────────────────────

    # If live results are available, use them
    type_map = {
        "Dueling-DQN": "Deep RL", "Double-DQN": "Deep RL",
        "DQN": "Deep RL", "PPO": "PPO",
        "Q-Learning": "Tabular RL", "SARSA": "Tabular RL",
        "Expected-SARSA": "Tabular RL",
        "Greedy": "Heuristic", "Best-Deal": "Heuristic",
        "TimeAware": "Heuristic", "Prof-Heuristic": "Heuristic",
        "Freshness-1st": "Heuristic", "Random": "Random",
    }
    if results:
        def _reward(r):
            return r.get('total_reward_mean', r.get('reward', 0))
        agents_data = [
            (name, _reward(r), type_map.get(name, "Heuristic"))
            for name, r in results.items()
        ]
        agents_data.sort(key=lambda x: x[1], reverse=True)
    else:
        agents_data = agents_static

    names   = [a[0] for a in agents_data]
    rewards = [a[1] for a in agents_data]
    colors  = [CAT_COLORS[a[2]] for a in agents_data]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG)

    bars = ax.barh(names, rewards, color=colors, height=0.65, zorder=3)

    for bar, val in zip(bars, rewards):
        x_pos = val + 5 if val >= 0 else val - 5
        ha    = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va='center', ha=ha,
                fontsize=10, fontweight='bold')

    ax.axvline(0, color='#333', lw=1.2, zorder=4)
    ax.set_xlabel("Mean Cumulative Reward (1,000 test episodes)", fontsize=LABEL_FS)
    ax.set_xlim(-130, 580)
    ax.grid(axis='x', color=GRID_CLR, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in CAT_COLORS.items()]
    ax.legend(handles=legend_patches, loc='lower right',
              fontsize=9, framealpha=0.9)

    ax.set_title(
        "Mean Cumulative Reward — All 12 Agents\n"
        "(Baseline Environment, 1,000 Test Episodes)",
        fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    _save(fig, "fig2_reward_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Radar / Spider chart — Six-Metric Performance Profile
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig3(results=None):
    # ── DATA ───────────────────────────────────────────────────────────────────
    axes_labels = [
        "Completion\n%", "Reward\n(norm)", "Product\nFreshness",
        "Brand\nQuality", "Price\nDiscount", "Routing\nEfficiency",
    ]
    # Values are 0–1 normalised. Order matches axes_labels above.
    # [completion, reward_norm, freshness_norm, quality_norm, discount_norm, routing_eff]
    agents_radar = {
        "Dueling-DQN":   [0.983, 1.00, 0.64, 0.63, 1.00, 1.00],
        "DQN":           [0.987, 0.95, 0.63, 0.62, 0.97, 0.98],
        "SARSA":         [0.959, 0.76, 0.58, 0.60, 0.52, 0.49],
        "TimeAware":     [1.000, 0.51, 0.62, 1.00, 0.19, 0.43],
        "Prof-Heuristic":[0.949, 0.30, 0.00, 0.97, 0.05, 0.40],
    }
    agent_colors_radar = {
        "Dueling-DQN":    "#1f77b4",
        "DQN":            "#aec7e8",
        "SARSA":          "#d95f02",
        "TimeAware":      "#2ca02c",
        "Prof-Heuristic": "#9467bd",
    }
    # ──────────────────────────────────────────────────────────────────────────

    N      = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True), facecolor=BG)
    ax.set_facecolor(BG)

    for name, vals in agents_radar.items():
        v     = vals + vals[:1]
        color = agent_colors_radar[name]
        ax.plot(angles, v, color=color, lw=2, marker='o',
                markersize=5, label=name)
        ax.fill(angles, v, color=color, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10, fontweight='bold')
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"],
                       fontsize=8, color='#666')
    ax.set_ylim(0, 1)
    ax.yaxis.grid(color=GRID_CLR, linestyle='--', alpha=0.6)
    ax.xaxis.grid(color='#999', linestyle='-', alpha=0.4)
    ax.spines['polar'].set_color('#333')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.9)

    ax.set_title(
        "Normalised Six-Metric Performance Profile\n"
        "(Selected Agents — Higher = Better on Each Axis)",
        fontsize=TITLE_FS, fontweight='bold', pad=20)
    plt.tight_layout()
    _save(fig, "fig3_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Mean Reward Across All 4 Parameter Sensitivity Experiments
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig4(exp_results=None):
    # ── DATA ───────────────────────────────────────────────────────────────────
    exp_labels = [
        "Baseline\n(120min 60%)",
        "Exp1\n(180min revisit)",
        "Exp2\n(10 goods/store)",
        "Exp3\n(80% avail)",
    ]
    # agent: [baseline, exp1, exp2, exp3]
    agent_data = {
        "Double-DQN": [485.12, 444.30,  360.02, 523.78],
        "DQN":        [472.93, 555.10,  313.82, 567.29],
        "Q-Learning": [395.00, 292.90,  294.42, 475.29],
        "PPO":        [217.40, 118.41, -101.69, 276.66],
        "Greedy":     [301.64, 302.92,  317.93, 366.38],
        "TimeAware":  [254.37, 256.60,  177.98, 261.75],
    }
    agent_colors_bar = {
        "Double-DQN": "#1f77b4",
        "DQN":        "#aec7e8",
        "Q-Learning": "#d95f02",
        "PPO":        "#e6ac00",
        "Greedy":     "#2ca02c",
        "TimeAware":  "#006e5f",
    }
    # ──────────────────────────────────────────────────────────────────────────

    # Override with live experiment results if available
    if exp_results:
        exp_keys = ["baseline", "Exp1_180min_Revisit",
                    "Exp2_10Goods_Store", "Exp3_80pct_Avail"]
        for agent in list(agent_data.keys()):
            row = []
            for key in exp_keys:
                if key in exp_results and agent in exp_results[key]:
                    row.append(exp_results[key][agent].get(
                        'total_reward_mean', agent_data[agent][len(row)]))
                else:
                    row.append(agent_data[agent][len(row)])
            agent_data[agent] = row

    n_agents = len(agent_data)
    n_exps   = len(exp_labels)
    x        = np.arange(n_exps)
    width    = 0.13
    offsets  = np.linspace(-(n_agents - 1) / 2,
                            (n_agents - 1) / 2, n_agents) * width

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)

    for i, (agent, vals) in enumerate(agent_data.items()):
        ax.bar(x + offsets[i], vals, width=width * 0.9,
               color=agent_colors_bar[agent], label=agent, zorder=3)

    ax.axhline(0, color='#555', lw=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=10)
    ax.set_ylabel("Mean Cumulative Reward", fontsize=LABEL_FS)
    ax.grid(axis='y', color=GRID_CLR, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=8.5, loc='upper left', framealpha=0.9)
    ax.set_title(
        " Mean Reward Across All Four Parameter Sensitivity Experiments",
        fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    _save(fig, "fig4_experiments.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Scatter: Completion Rate vs Routing Efficiency (bubble chart)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig5(results=None):
    # ── DATA ───────────────────────────────────────────────────────────────────
    # (name, stores_visited, completion_pct, reward, hex_color)
    agents_scatter_static = [
        ("DQN",            2.47,  98.7,  472.93, "#1f77b4"),
        ("Dueling-DQN",    2.55,  98.3,  498.66, "#1f77b4"),
        ("Double-DQN",     2.71,  97.1,  475.40, "#1f77b4"),
        ("Q-Learning",     4.50,  91.5,  355.00, "#d95f02"),
        ("SARSA",          4.80,  95.9,  375.77, "#d95f02"),
        ("Best-Deal",      5.50,  99.9,  338.47, "#2ca02c"),
        ("Greedy",         6.00, 100.0,  301.64, "#2ca02c"),
        ("TimeAware",      5.60, 100.0,  254.37, "#2ca02c"),
        ("Freshness-1st",  5.20, 100.0,  152.59, "#2ca02c"),
        ("Prof-Heuristic", 5.35,  94.9,  148.69, "#9467bd"),
        ("PPO",            5.21,  99.6,  184.17, "#e6ac00"),
        ("Random",         3.41,  62.8,  -50.88, "#888888"),
    ]
    color_map = {
        "Dueling-DQN": "#1f77b4", "Double-DQN": "#1f77b4", "DQN": "#1f77b4",
        "SARSA": "#d95f02", "Q-Learning": "#d95f02", "Expected-SARSA": "#d95f02",
        "Greedy": "#2ca02c", "Best-Deal": "#2ca02c", "TimeAware": "#2ca02c",
        "Freshness-1st": "#2ca02c", "Prof-Heuristic": "#9467bd",
        "PPO": "#e6ac00", "Random": "#888888",
    }
    # Label nudges (x_offset, y_offset) for avoiding overlaps
    label_nudge = {
        "DQN":            (-0.05,  0.6),
        "Dueling-DQN":    ( 0.04,  0.6),
        "Double-DQN":     ( 0.04, -1.0),
        "Prof-Heuristic": ( 0.04, -1.2),
        "PPO":            ( 0.00,  0.7),
        "Freshness-1st":  (-0.35,  0.7),
        "Best-Deal":      ( 0.04,  0.7),
    }
    # ──────────────────────────────────────────────────────────────────────────

    if results:
        def _get(r, *keys, default=0):
            for k in keys:
                if k in r: return r[k]
            return default
        agents_scatter = [
            (name,
             _get(r, 'stores_visited_mean', 'stores'),
             _get(r, 'completion_rate_mean', 'completion') * (1 if _get(r,'completion_rate_mean','completion') <= 1 else 0.01) * 100
             if _get(r, 'completion_rate_mean', r.get('completion', 1)) <= 1
             else _get(r, 'completion_rate_mean', r.get('completion', 100)),
             _get(r, 'total_reward_mean', 'reward'),
             color_map.get(name, "#888888"))
            for name, r in results.items()
        ]
    else:
        agents_scatter = agents_scatter_static

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(BG)

    for name, stores, comp, reward, color in agents_scatter:
        size = max(30, (reward + 100) * 0.6)
        ax.scatter(stores, comp, s=size, color=color, alpha=0.85,
                   edgecolors='white', linewidths=0.8, zorder=4)
        dx, dy = label_nudge.get(name, (0.04, 0.6))
        ax.text(stores + dx, comp + dy, name, fontsize=8.5, va='bottom')

    ax.axvline(4.0, color='#e74c3c', lw=1, linestyle='--', alpha=0.6)
    ax.axhline(95,  color='#e74c3c', lw=1, linestyle='--', alpha=0.6)
    ax.text(1.55, 101.5,
            "← Fewer stores\n   = more efficient",
            fontsize=8.5, color='#2980b9')
    ax.set_xlabel("Mean Stores Visited per Episode",  fontsize=LABEL_FS)
    ax.set_ylabel("Shopping Completion Rate (%)",     fontsize=LABEL_FS)
    ax.set_xlim(1.8, 6.8)
    ax.set_ylim(55, 105)
    ax.grid(color=GRID_CLR, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Completion Rate vs Routing Efficiency\n"
        "(Bubble size = Mean Reward.  Ideal agent: top-left corner)",
        fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    _save(fig, "fig5_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Price Discount vs Brand Quality Trade-off
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig6(results=None):
    # ── DATA ───────────────────────────────────────────────────────────────────
    # (name, price_premium_pct, brand_pref_score_out_of_5, hex_color)
    agents_pq_static = [
        ("Dueling-DQN",    -17.10, 3.43, "#1f77b4"),
        ("Double-DQN",     -15.80, 3.39, "#1f77b4"),
        ("DQN",            -16.20, 3.41, "#1f77b4"),
        ("SARSA",           -8.74, 3.31, "#d95f02"),
        ("Q-Learning",      -8.90, 3.35, "#d95f02"),
        ("Greedy",          -6.93, 3.45, "#2ca02c"),
        ("Best-Deal",       -0.95, 3.25, "#2ca02c"),
        ("TimeAware",       -3.20, 4.49, "#17becf"),
        ("PPO",             -2.10, 3.48, "#e6ac00"),
        ("Freshness-1st",   -1.80, 3.51, "#8c564b"),
        ("Prof-Heuristic",  -0.95, 4.17, "#9467bd"),
        ("Random",          -0.45, 3.02, "#888888"),
    ]
    color_map6 = {
        "Dueling-DQN": "#1f77b4", "Double-DQN": "#1f77b4", "DQN": "#1f77b4",
        "SARSA": "#d95f02", "Q-Learning": "#d95f02",
        "Greedy": "#2ca02c", "Best-Deal": "#2ca02c",
        "TimeAware": "#17becf", "PPO": "#e6ac00",
        "Freshness-1st": "#8c564b", "Prof-Heuristic": "#9467bd",
        "Random": "#888888",
    }
    label_nudge6 = {
        "Dueling-DQN": (-0.3,  0.02),
        "Double-DQN":  (-0.3, -0.05),
        "DQN":         ( 0.2,  0.02),
        "Q-Learning":  ( 0.15, 0.02),
        "SARSA":       ( 0.15,-0.05),
        "PPO":         (-0.5,  0.04),
        "Freshness-1st":(-0.2, 0.05),
    }
    # ──────────────────────────────────────────────────────────────────────────

    if results:
        def _get6(r, *keys, default=0):
            for k in keys:
                if k in r: return r[k]
            return default
        agents_pq = [
            (name,
             _get6(r, 'avg_premium_pct_mean', 'premium'),
             _get6(r, 'avg_pref_score_mean', 'pref'),
             color_map6.get(name, "#888888"))
            for name, r in results.items()
        ]
    else:
        agents_pq = agents_pq_static

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(BG)

    for name, prem, pref, color in agents_pq:
        ax.scatter(prem, pref, s=160, color=color, alpha=0.85,
                   edgecolors='white', lw=0.8, zorder=4)
        dx, dy = label_nudge6.get(name, (0.15, 0.02))
        ax.text(prem + dx, pref + dy, name, fontsize=8.5)

    ax.axvline(-5,  color='#e74c3c', lw=1, linestyle='--', alpha=0.5)
    ax.axhline(3.5, color='#e74c3c', lw=1, linestyle='--', alpha=0.5)

    ax.text(-18.5, 4.54, "Best quality\nhigh discount",
            fontsize=8, color='#155724',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda',
                      edgecolor='#28a745', lw=1.5))
    ax.text(-1.5, 3.07, "Low discount\nlow quality",
            fontsize=8, color='#721c24',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8d7da',
                      edgecolor='#dc3545', lw=1.5))

    ax.set_xlabel("Mean Price Premium % (more negative = cheaper)", fontsize=LABEL_FS)
    ax.set_ylabel("Mean Brand Preference Score (out of 5)",         fontsize=LABEL_FS)
    ax.grid(color=GRID_CLR, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Price Discount vs Brand Quality Trade-off\n"
        "(Ideal: bottom-left = cheap prices, top = preferred brands)",
        fontsize=TITLE_FS, fontweight='bold')
    plt.tight_layout()
    _save(fig, "fig6_price_quality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — DQN Neural Network Architecture diagram
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig7():
    # ── DATA ───────────────────────────────────────────────────────────────────
    layers = [
        {"size": 393, "label": "393", "sublabel": "Input\nState Vector",
         "color": "#1a3a5c", "w": 0.14},
        {"size": 64,  "label": "64",  "sublabel": "Hidden\nLayer 1",
         "color": "#2176ae", "w": 0.09},
        {"size": 32,  "label": "32",  "sublabel": "Hidden\nLayer 2",
         "color": "#57c4e5", "w": 0.07},
        {"size": 107, "label": "107", "sublabel": "Q-Values\nOutput",
         "color": "#3cb371", "w": 0.11},
    ]
    activations = ["ReLU", "ReLU", "Linear (no activation)"]
    left_annotations = [
        "6: store one-hot",
        "1: time/120",
        "20: items needed",
        "6: visited stores",
        "360: revealed info",
    ]
    right_annotations = [
        "1: End",
        "100: Buy(good×brand)",
        "6: Travel(store)",
    ]
    # ──────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)

    x_positions = [1.5, 4.5, 7.2, 10.5]
    max_h       = 3.8

    for i, (layer, xp) in enumerate(zip(layers, x_positions)):
        h = max(0.5, (layer["size"] / 393) * max_h)
        ax.add_patch(plt.Rectangle(
            (xp - layer["w"] * 5, (5 - h) / 2),
            layer["w"] * 10, h,
            color=layer["color"], zorder=3, linewidth=0))
        ax.text(xp, 2.5, layer["label"], ha='center', va='center',
                fontsize=16, fontweight='bold', color='white', zorder=4)
        ax.text(xp, (5 - h) / 2 - 0.35, layer["sublabel"],
                ha='center', va='top', fontsize=9, color='#333')

        if i < len(layers) - 1:
            x_next = x_positions[i + 1]
            ax.annotate("",
                        xy=(x_next - layers[i+1]["w"] * 5, 2.5),
                        xytext=(xp + layer["w"] * 5, 2.5),
                        arrowprops=dict(arrowstyle="-|>", color='#333',
                                       lw=1.5, mutation_scale=18))
            ax.text((xp + x_next) / 2, 3.1, activations[i],
                    ha='center', va='bottom', fontsize=9,
                    color='#c0392b', style='italic')

    ax.text(0.15, 4.5, "State components:",
            fontsize=9, fontweight='bold')
    for j, ann in enumerate(left_annotations):
        ax.text(0.15, 4.05 - j * 0.55, ann, fontsize=8.5, color='#555')

    ax.text(12.85, 4.5, "Action components:",
            fontsize=9, fontweight='bold', ha='right')
    for j, ann in enumerate(right_annotations):
        ax.text(12.85, 4.05 - j * 0.7, ann,
                fontsize=8.5, color='#555', ha='right')

    ax.set_title(
        " DQN Neural Network Architecture (393 → 64 → 32 → 107)",
        fontsize=TITLE_FS, fontweight='bold', pad=10)
    plt.tight_layout()
    _save(fig, "fig7_dqn_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Tabular Agent Convergence Curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fig8(training_curves=None):
    # ── DATA ───────────────────────────────────────────────────────────────────
    episodes_k = [0, 10, 25, 40, 60, 80, 100, 120, 150, 200]

    completion_static = {
        "Q-Learning":     [50, 65, 75, 86, 91, 93, 94,   94.5, 91,  91.5],
        "SARSA":          [50, 65, 76, 86, 92, 94, 95,   95.5, 95,  95.0],
        "Expected-SARSA": [50, 60, 70, 81, 87, 90, 91,   92.0, 92,  92.0],
    }
    reward_static = {
        "Q-Learning":     [-50,  80, 175, 250, 310, 320, 330, 335, 330, 335],
        "SARSA":          [-50,  85, 180, 255, 315, 340, 350, 355, 355, 358],
        "Expected-SARSA": [-50,  45, 125, 225, 305, 315, 320, 325, 325, 330],
    }
    agent_colors_curve = {
        "Q-Learning":     "#e6ac00",
        "SARSA":          "#d95f02",
        "Expected-SARSA": "#1f77b4",
    }
    # ──────────────────────────────────────────────────────────────────────────

    # If live training curves passed in, use them
    if training_curves and any(k in training_curves
                               for k in ["Q-Learning", "SARSA", "Expected-SARSA"]):
        completion_data = {}
        reward_data     = {}
        for agent in ["Q-Learning", "SARSA", "Expected-SARSA"]:
            if agent in training_curves:
                tc = training_curves[agent]
                n  = len(tc['rewards'])
                step = max(1, n // 10)
                reward_data[agent]     = [float(v) for v in tc['rewards'][::step]][:10]
                completion_data[agent] = [float(v) * 100
                                          for v in tc['completion'][::step]][:10]
                # Pad to 10 if shorter
                while len(reward_data[agent]) < 10:
                    reward_data[agent].append(reward_data[agent][-1])
                    completion_data[agent].append(completion_data[agent][-1])
            else:
                completion_data[agent] = completion_static[agent]
                reward_data[agent]     = reward_static[agent]
    else:
        completion_data = completion_static
        reward_data     = reward_static

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)

    for ax, data, ylabel, title in [
        (ax1, completion_data, "Shopping Completion Rate (%)",
         "Completion Rate vs Training Episodes"),
        (ax2, reward_data,     "Mean Cumulative Reward",
         "Mean Reward vs Training Episodes"),
    ]:
        ax.set_facecolor(BG)
        for agent, vals in data.items():
            ep = episodes_k[:len(vals)]
            ax.plot(ep, vals, color=agent_colors_curve[agent],
                    lw=2.2, marker='o', markersize=5, label=agent)

        ax.axvline(10, color='#e74c3c', lw=1, linestyle='--', alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(10.5, ylim[0] + (ylim[1] - ylim[0]) * 0.05,
                "10k\n(initial run)", fontsize=7.5, color='#e74c3c')
        ax.set_xlabel("Training Episodes (thousands)", fontsize=LABEL_FS)
        ax.set_ylabel(ylabel,                          fontsize=LABEL_FS)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(color=GRID_CLR, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)

    fig.suptitle(
        " Tabular Agent Convergence Curves (200,000 Training Episodes)",
        fontsize=TITLE_FS, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "fig8_convergence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — called from main.py
# ═══════════════════════════════════════════════════════════════════════════════
def generate_all_figures(results=None, exp_results=None, training_curves=None):
    """
    Generate all 8 figures.

    Parameters
    ----------
    results : dict, optional
        Output from evaluate_agent() for each agent (from main.py).
        If None, hardcoded dissertation values are used.
    exp_results : dict, optional
        Nested dict of experiment results (from run_experiments.py).
        Keys: 'baseline', 'Exp1_180min_Revisit', etc.
    training_curves : dict, optional
        Training history from train_agent() (from main.py).
        Used for Fig 8 convergence curves.
    """
    print("\n── Generating figures ─────────────────────────────────────")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    plot_fig1()                                  # static diagram — no data needed
    plot_fig2(results=results)                   # uses live results if available
    plot_fig3(results=results)                   # normalised radar
    plot_fig4(exp_results=exp_results)           # experiment comparison bars
    plot_fig5(results=results)                   # bubble scatter
    plot_fig6(results=results)                   # price vs quality
    plot_fig7()                                  # static architecture diagram
    plot_fig8(training_curves=training_curves)   # convergence curves

    print(f"── All figures saved to: {os.path.abspath(FIGURES_DIR)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dissertation figures from results JSON")
    parser.add_argument("--data", default=None,
                        help="Path to results.json (default: auto-detect)")
    args = parser.parse_args()

    results        = None
    exp_results    = None
    training_curves = None

    # Try loading results.json
    candidates = [
        args.data,
        os.path.join(os.path.dirname(__file__), "results.json"),
        os.path.join(os.path.dirname(__file__), "..", "master_results.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # results.json from main.py has a 'results' key
            results         = data.get('results', data)
            training_curves = data.get('training_curves', None)
            print(f"  Loaded results from: {path}")
            break

    # Try loading experiment_results.json
    exp_candidates = [
        os.path.join(os.path.dirname(__file__), "experiment_results.json"),
        os.path.join(os.path.dirname(__file__), "..", "all_experiment_results.json"),
    ]
    for path in exp_candidates:
        if os.path.exists(path):
            with open(path) as f:
                exp_data    = json.load(f)
            exp_results = exp_data.get('experiments',
                          exp_data.get('results', exp_data))
            print(f"  Loaded experiment results from: {path}")
            break

    if not results:
        print("  No results.json found — using hardcoded dissertation values")

    generate_all_figures(results=results,
                         exp_results=exp_results,
                         training_curves=training_curves)
