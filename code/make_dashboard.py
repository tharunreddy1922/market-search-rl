"""
Standalone dashboard generator — run this instead of the full experiments.py
Usage:  python make_dashboard.py
Output: dashboard_fixed.html  (open directly in any browser, no server needed)
"""

import json
import os
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = CODE_DIR   # adjust if your JSON files are elsewhere
OUT_PATH    = os.path.join(CODE_DIR, '..', 'dashboard_fixed.html')

# ── numpy-safe JSON encoder ────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def jdump(obj):
    return json.dumps(obj, cls=NumpyEncoder)

# ── load JSON files ────────────────────────────────────────────────────────────
def load(name):
    p = os.path.join(RESULTS_DIR, name)
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    print(f'  [warn] {name} not found')
    return {}

print('Loading JSON files...')
experiment_results = load('experiment_results.json')
training_size      = load('training_size_results.json')
pg_sweep           = load('pg_sweep_results.json')
full_results       = load('full_experiment_results.json')

# ── build results dict ─────────────────────────────────────────────────────────
results = dict(experiment_results)
if training_size:
    results['training_size'] = training_size

# sweep dict (for convergence tab)
sweep = {}
if training_size:
    sweep['tabular'] = training_size.get('tabular', {})
    sweep['deep']    = training_size.get('deep', {})
if full_results:
    if 'tabular' not in sweep:
        sweep['tabular'] = {}
    if 'deep' not in sweep:
        sweep['deep'] = {}

# ── constants ──────────────────────────────────────────────────────────────────
TEST_EPISODES = 1000

AGENTS_ORDERED = [
    'Q-Learning','SARSA','ExpSARSA',
    'DQN','DoubleDQN','DuelingDQN',
    'PPO','REINFORCE','A2C',
    'ValueIter',
    'Greedy','TimeAware','Freshness','BrandQuality',
    'Random',
]

AGENT_COLORS = {
    'Q-Learning' :'#f05050','SARSA'      :'#f07850','ExpSARSA'   :'#f0a050',
    'DQN'        :'#5080f0','DoubleDQN'  :'#7c83fd','DuelingDQN' :'#50c8f0',
    'PPO'        :'#a050f0','REINFORCE'  :'#c050d0','A2C'        :'#d050a0',
    'ValueIter'  :'#50d0b0',
    'Greedy'     :'#52c77a','TimeAware'  :'#80d050','Freshness'  :'#c8e050',
    'BrandQuality':'#e0c050','Random'    :'#888888',
}

AGENT_GROUPS = {
    'Tabular RL'   : ['Q-Learning','SARSA','ExpSARSA'],
    'Deep RL'      : ['DQN','DoubleDQN','DuelingDQN'],
    'Policy Grad.' : ['PPO','REINFORCE','A2C'],
    'Model-Based'  : ['ValueIter'],
    'Heuristic'    : ['Greedy','TimeAware','Freshness','BrandQuality'],
    'Baseline'     : ['Random'],
}

# ── serialise data for embedding ───────────────────────────────────────────────
R_json        = jdump(results)
SW_json       = jdump(sweep)
AG_json       = jdump(AGENTS_ORDERED)
C_json        = jdump(AGENT_COLORS)
GROUPS_json   = jdump(AGENT_GROUPS)
N             = TEST_EPISODES

# ── build HTML (Chart.js embedded inline so no CDN needed) ────────────────────
# We download Chart.js source inline via a data URI trick —
# simpler: just use the unpkg fallback and also a local copy comment.
# Actually safest: embed a minimal stub. But for full charts we use the CDN
# with crossOrigin removed and a fallback notice.

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Market Search RL — Experiment Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"
        crossorigin="anonymous"
        referrerpolicy="no-referrer"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f1117;color:#e8eaf6;font-family:'Segoe UI',sans-serif;font-size:14px}}
  h1{{padding:22px 32px 6px;font-size:20px;color:#7c83fd}}
  .sub{{padding:0 32px 18px;color:#9fa8da;font-size:12px}}
  .tabs{{display:flex;gap:4px;padding:0 32px;border-bottom:1px solid #1e2030;flex-wrap:wrap}}
  .tab{{padding:9px 18px;cursor:pointer;border-radius:8px 8px 0 0;color:#9fa8da;
        background:#1a1d2e;font-size:12px;transition:all .2s}}
  .tab.active,.tab:hover{{background:#7c83fd;color:#fff}}
  .panel{{display:none;padding:24px 32px}}
  .panel.active{{display:block}}
  .kpis{{display:flex;gap:14px;margin-bottom:24px;flex-wrap:wrap}}
  .kpi{{background:#1a1d2e;border-radius:10px;padding:14px 20px;flex:1;min-width:140px}}
  .kv{{font-size:24px;font-weight:700;color:#7c83fd}}
  .kl{{font-size:11px;color:#9fa8da;margin-top:3px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin-bottom:20px}}
  .box{{background:#1a1d2e;border-radius:10px;padding:18px}}
  .box h3{{font-size:12px;color:#9fa8da;margin-bottom:12px}}
  canvas{{max-height:260px}}
  table{{width:100%;border-collapse:collapse;font-size:12px;background:#1a1d2e;border-radius:8px;overflow:hidden}}
  th{{background:#13152b;color:#9fa8da;padding:9px 11px;text-align:left;font-weight:600}}
  td{{padding:8px 11px;border-bottom:1px solid #1e2030}}
  tr:hover td{{background:#1f2340}}
  .g{{color:#52c77a;font-weight:600}} .a{{color:#f0a500;font-weight:600}} .r{{color:#f05050;font-weight:600}}
  .r1{{color:#ffd700;font-weight:700}} .r2{{color:#c0c0c0;font-weight:700}} .r3{{color:#cd7f32;font-weight:700}}
  .sig{{font-size:11px;padding:1px 5px;border-radius:4px;background:#1b3a1b;color:#52c77a;margin-left:4px}}
  .ns{{font-size:11px;padding:1px 5px;border-radius:4px;background:#2a1010;color:#f05050;margin-left:4px}}
  #chartjs-warn{{display:none;background:#3a2010;color:#f0a050;padding:12px 24px;font-size:12px}}
</style>
</head>
<body>
<div id="chartjs-warn">
  ⚠ Chart.js failed to load from CDN. Open this file via a local server:<br>
  &nbsp;&nbsp;Run: <code>python -m http.server 8080</code> then visit
  <a href="http://127.0.0.1:8080/dashboard_fixed.html" style="color:#7c83fd">
  http://127.0.0.1:8080/dashboard_fixed.html</a>
</div>
<h1>Market Search and Purchase Scheduling — RL Experiment Dashboard</h1>
<p class="sub">MSc Data Science | University of Roehampton | A00051705<br>
N={N} test episodes (fixed seeds) | 95% CI on all metrics | Welch's t-test significance</p>
<div class="tabs">
  <div class="tab active" onclick="show('overview',this)">Overview</div>
  <div class="tab" onclick="show('baseline',this)">Baseline + CI</div>
  <div class="tab" onclick="show('oracle',this)">Oracle Gap</div>
  <div class="tab" onclick="show('policy',this)">Policy Analysis</div>
  <div class="tab" onclick="show('graph',this)">Graph Experiments</div>
  <div class="tab" onclick="show('brand',this)">Brand Experiments</div>
  <div class="tab" onclick="show('param',this)">Parametric</div>
  <div class="tab" onclick="show('sig',this)">Significance</div>
  <div class="tab" onclick="show('conv',this)">Convergence</div>
  <div class="tab" onclick="show('rank',this)">Rankings</div>
</div>

<div id="tab-overview" class="panel active">
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="box"><h3>Mean Reward — all agents</h3><canvas id="c-ov-r"></canvas></div>
    <div class="box"><h3>Completion Rate — all agents</h3><canvas id="c-ov-cr"></canvas></div>
  </div>
</div>

<div id="tab-baseline" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    95% confidence intervals: 1.96 x sigma / sqrt(n), n={N} episodes.
  </p>
  <div class="grid">
    <div class="box"><h3>Premium vs Preference Score</h3><canvas id="c-bl-pp"></canvas></div>
    <div class="box"><h3>Avg Expiry Days Achieved</h3><canvas id="c-bl-ex"></canvas></div>
  </div>
  <table id="t-baseline"></table>
</div>

<div id="tab-oracle" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Oracle has perfect information (availability, expiry, prices) before each episode.
    Gap = 100 x (R_oracle - R_agent) / |R_oracle| per episode.
  </p>
  <div class="grid">
    <div class="box"><h3>Optimality Gap (%)</h3><canvas id="c-or-gap"></canvas></div>
    <div class="box"><h3>Reward vs Oracle</h3><canvas id="c-or-r"></canvas></div>
  </div>
  <table id="t-oracle"></table>
</div>

<div id="tab-policy" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Store-visit frequencies and route patterns across {N} test episodes.
  </p>
  <div class="grid">
    <div class="box"><h3>Store Visit Frequency by Agent</h3><canvas id="c-pol-freq"></canvas></div>
    <div class="box"><h3>Avg Route Length by Agent</h3><canvas id="c-pol-len"></canvas></div>
  </div>
  <table id="t-policy"></table>
  <br>
  <table id="t-store-chars"></table>
</div>

<div id="tab-graph" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Graph structure experiments: baseline (all 5 min) vs one/two distant stores (20 min).
  </p>
  <div class="grid">
    <div class="box"><h3>Reward vs Graph Configuration</h3><canvas id="c-gph-r"></canvas></div>
    <div class="box"><h3>Completion Rate vs Graph Configuration</h3><canvas id="c-gph-cr"></canvas></div>
  </div>
  <table id="t-graph"></table>
</div>

<div id="tab-brand" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Brand configuration experiments: narrow (5B/2), baseline (5B/3), rich (10B/3).
  </p>
  <div class="grid">
    <div class="box"><h3>Reward vs Brand Configuration</h3><canvas id="c-br-r"></canvas></div>
    <div class="box"><h3>Completion Rate vs Brand Configuration</h3><canvas id="c-br-cr"></canvas></div>
  </div>
  <table id="t-brand"></table>
</div>

<div id="tab-param" class="panel">
  <div class="grid">
    <div class="box"><h3>Completion Rate vs Duration</h3><canvas id="c-dur-cr"></canvas></div>
    <div class="box"><h3>Completion Rate vs Availability</h3><canvas id="c-av-cr"></canvas></div>
    <div class="box"><h3>Completion Rate vs Goods Count</h3><canvas id="c-gd-cr"></canvas></div>
  </div>
  <table id="t-param"></table>
</div>

<div id="tab-sig" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Welch's t-test (two-tailed, alpha=0.05). Cohen's d: small &lt;0.2, medium 0.2-0.8, large &gt;0.8.
  </p>
  <table id="t-sig"></table>
</div>

<div id="tab-conv" class="panel">
  <div class="grid">
    <div class="box"><h3>Tabular — Reward vs Training Size</h3><canvas id="c-tab-r"></canvas></div>
    <div class="box"><h3>Deep RL — Reward vs Training Size</h3><canvas id="c-deep-r"></canvas></div>
  </div>
  <table id="t-conv"></table>
</div>

<div id="tab-rank" class="panel">
  <table id="t-rank"></table>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────────────────
const R  = {R_json};
const SW = {SW_json};
const AG = {AG_json};
const C  = {C_json};
const GROUPS = {GROUPS_json};

// ── check Chart.js loaded ─────────────────────────────────────────────────────
if (typeof Chart === 'undefined') {{
  document.getElementById('chartjs-warn').style.display = 'block';
}}

// ── helpers ───────────────────────────────────────────────────────────────────
function show(id, el) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
}}

function rgba(h, a) {{
  if (!h || h.length < 7) return 'rgba(124,131,253,' + a + ')';
  const r = parseInt(h.slice(1,3),16);
  const g = parseInt(h.slice(3,5),16);
  const b = parseInt(h.slice(5,7),16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
}}

function n(v, d) {{
  if (d === undefined) d = 1;
  return (v == null || isNaN(v)) ? '—' : Number(v).toFixed(d);
}}

function cr(v) {{
  if (v == null) return '—';
  const cl = v >= 0.95 ? 'g' : v >= 0.80 ? 'a' : 'r';
  return '<span class="' + cl + '">' + (v * 100).toFixed(1) + '%</span>';
}}

const defs = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{ legend: {{ labels: {{ color: '#e8eaf6', font: {{ size: 10 }}, boxWidth: 10 }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#9fa8da' }}, grid: {{ color: 'rgba(255,255,255,.05)' }} }},
    y: {{ ticks: {{ color: '#9fa8da' }}, grid: {{ color: 'rgba(255,255,255,.08)' }} }}
  }}
}};

function bar(id, labels, ds, yO) {{
  if (typeof Chart === 'undefined') return;
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const yOpts = yO || {{}};
  new Chart(ctx.getContext('2d'), {{
    type: 'bar',
    data: {{ labels: labels, datasets: ds }},
    options: {{ ...defs, scales: {{ ...defs.scales, y: {{ ...defs.scales.y, ...yOpts }} }} }}
  }});
}}

function line(id, labels, ds, yO) {{
  if (typeof Chart === 'undefined') return;
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const yOpts = yO || {{}};
  new Chart(ctx.getContext('2d'), {{
    type: 'line',
    data: {{ labels: labels, datasets: ds }},
    options: {{ ...defs, scales: {{ ...defs.scales, y: {{ ...defs.scales.y, ...yOpts }} }} }}
  }});
}}

// ── baseline ──────────────────────────────────────────────────────────────────
const bl    = R.baseline || {{}};
const names = AG.filter(a => bl[a]);
const rewards = names.map(a => bl[a].mean_reward);
const best  = names[rewards.indexOf(Math.max(...rewards))];

document.getElementById('kpis').innerHTML =
  '<div class="kpi"><div class="kv">' + best + '</div><div class="kl">Best agent (reward)</div></div>' +
  '<div class="kpi"><div class="kv">' + n(Math.max(...rewards), 0) + '</div><div class="kl">Peak reward +/-' + n(bl[best] && bl[best].ci95_reward, 1) + '</div></div>' +
  '<div class="kpi"><div class="kv">{N}</div><div class="kl">Test episodes (95% CI)</div></div>' +
  '<div class="kpi"><div class="kv">' + names.length + '</div><div class="kl">Algorithms compared</div></div>';

bar('c-ov-r', names, [{{ label: 'Reward', data: names.map(a => bl[a].mean_reward),
  backgroundColor: names.map(a => rgba(C[a] || '#7c83fd', .8)),
  borderColor: names.map(a => C[a] || '#7c83fd'), borderWidth: 1 }}]);

bar('c-ov-cr', names, [{{ label: 'Completion', data: names.map(a => bl[a].mean_cr),
  backgroundColor: names.map(a => rgba(C[a] || '#7c83fd', .8)),
  borderColor: names.map(a => C[a] || '#7c83fd'), borderWidth: 1 }}],
  {{ min: 0, max: 1.05, ticks: {{ callback: function(v) {{ return (v*100).toFixed(0)+'%'; }} }} }});

let h = '<tr><th>Agent</th><th>Reward</th><th>+/-CI</th><th>CR</th><th>+/-CI</th><th>Stores</th><th>Time</th><th>Premium%</th><th>Pref</th><th>Success</th></tr>';
names.forEach(function(a) {{
  const d = bl[a];
  h += '<tr><td><b>' + a + '</b></td><td>' + n(d.mean_reward,1) + '</td><td>+/-' + n(d.ci95_reward,1) + '</td>' +
       '<td>' + cr(d.mean_cr) + '</td><td>+/-' + n(d.ci95_cr,3) + '</td><td>' + n(d.mean_stores,1) + '</td>' +
       '<td>' + n(d.mean_time,0) + 'm</td><td>' + n(d.mean_premium,1) + '%</td>' +
       '<td>' + n(d.mean_pref,2) + '</td><td>' + n(d.success_rate*100,1) + '%</td></tr>';
}});
document.getElementById('t-baseline').innerHTML = h;

if (typeof Chart !== 'undefined') {{
  const ctx2 = document.getElementById('c-bl-pp');
  if (ctx2) new Chart(ctx2.getContext('2d'), {{
    type: 'scatter',
    data: {{ datasets: names.map(function(a) {{
      const d = bl[a];
      return {{ label: a, data: [{{ x: d.mean_premium, y: d.mean_pref }}],
               backgroundColor: rgba(C[a]||'#7c83fd', .85),
               pointRadius: 9, pointHoverRadius: 12 }};
    }}) }},
    options: {{ ...defs, scales: {{
      x: {{ ...defs.scales.x, title: {{ display: true, text: 'Avg Premium (%)', color: '#9fa8da' }} }},
      y: {{ ...defs.scales.y, title: {{ display: true, text: 'Avg Pref Score', color: '#9fa8da' }} }}
    }} }}
  }});
}}

bar('c-bl-ex', names, [{{ label: 'Expiry Days', data: names.map(a => (bl[a].mean_expiry||0)),
  backgroundColor: names.map(a => rgba(C[a]||'#7c83fd',.8)),
  borderColor: names.map(a => C[a]||'#7c83fd'), borderWidth: 1 }}]);

// ── oracle ────────────────────────────────────────────────────────────────────
const orb     = R.oracle_benchmark || {{}};
const ora     = orb.agents || {{}};
const ornames = Object.keys(ora);

bar('c-or-gap', ornames, [{{ label: 'Optimality Gap (%)',
  data: ornames.map(a => ora[a] && ora[a].optimality_gap_pct),
  backgroundColor: ornames.map(a => rgba(C[a]||'#7c83fd',.8)), borderWidth: 1 }}]);

const orvals = [{{ name: 'Oracle', mean_reward: (orb.oracle||{{}}).mean_reward }}]
  .concat(ornames.map(a => ({{ name: a, mean_reward: ora[a] && ora[a].mean_reward }})));
bar('c-or-r', orvals.map(v => v.name), [{{ label: 'Mean Reward',
  data: orvals.map(v => v.mean_reward),
  backgroundColor: orvals.map(v => rgba(C[v.name]||'#f1c40f',.8)), borderWidth: 1 }}]);

let ot = '<tr><th>Agent</th><th>Mean Reward</th><th>+/-CI</th><th>CR</th><th>Success Rate</th><th>Opt. Gap (%)</th><th>Gap Std</th></tr>';
const orc = orb.oracle || {{}};
ot += '<tr><td><b>Oracle</b></td><td>' + n(orc.mean_reward,1) + '</td><td>+/-' + n(orc.ci95_reward,1) +
      '</td><td>' + cr(orc.mean_cr) + '</td><td>' + n(orc.success_rate*100,1) + '%</td><td>0.0%</td><td>-</td></tr>';
ornames.forEach(function(a) {{
  const d = ora[a];
  ot += '<tr><td>' + a + '</td><td>' + n(d&&d.mean_reward,1) + '</td><td>+/-' + n(d&&d.ci95_reward,1) +
        '</td><td>' + cr(d&&d.mean_cr) + '</td><td>' + n(d&&d.success_rate*100,1) + '%</td>' +
        '<td>' + n(d&&d.optimality_gap_pct,1) + '%</td><td>+/-' + n(d&&d.gap_std_pct,1) + '%</td></tr>';
}});
document.getElementById('t-oracle').innerHTML = ot;

// ── policy analysis ───────────────────────────────────────────────────────────
const pol       = R.policy_analysis || {{}};
const polAgents = Object.keys(pol.agent_policies || {{}});
const storeLabels = ['Store 0','Store 1','Store 2','Store 3','Store 4','Store 5'];

if (polAgents.length > 0) {{
  const vf = polAgents.map(a => (pol.agent_policies[a] && pol.agent_policies[a].visit_freq) || []);
  const datasets = polAgents.map(function(a) {{
    return {{ label: a, data: pol.agent_policies[a] && pol.agent_policies[a].visit_freq || [],
             backgroundColor: rgba(C[a]||'#7c83fd',.7),
             borderColor: C[a]||'#7c83fd', borderWidth: 1 }};
  }});
  bar('c-pol-freq', storeLabels, datasets);
  bar('c-pol-len', polAgents, [{{ label: 'Avg Route Length',
    data: polAgents.map(a => pol.agent_policies[a] && pol.agent_policies[a].avg_route_length),
    backgroundColor: polAgents.map(a => rgba(C[a]||'#7c83fd',.8)), borderWidth: 1 }}]);

  let pt = '<tr><th>Agent</th><th>Dominant Store</th><th>Avg Route Length</th><th>Mean Reward</th><th>CR</th></tr>';
  polAgents.forEach(function(a) {{
    const d = pol.agent_policies[a];
    pt += '<tr><td><b>' + a + '</b></td><td>Store ' + (d&&d.dominant_store) + '</td>' +
          '<td>' + n(d&&d.avg_route_length,2) + '</td><td>' + n(d&&d.mean_reward,1) + '</td>' +
          '<td>' + cr(d&&d.mean_cr) + '</td></tr>';
  }});
  document.getElementById('t-policy').innerHTML = pt;

  const sc = pol.store_characteristics || [];
  let st = '<tr><th>Store</th><th>Avg Best Pref Score</th><th>Avg Price Premium (%)</th><th>Brands/Good</th><th>Travel Time (min)</th></tr>';
  sc.forEach(function(s) {{
    st += '<tr><td>Store ' + s.store_id + '</td><td>' + n(s.avg_best_pref_score,3) + '</td>' +
          '<td>' + n(s.avg_price_premium_pct,2) + '%</td><td>' + s.brands_stocked_per_good +
          '</td><td>' + s.travel_time_min + '</td></tr>';
  }});
  document.getElementById('t-store-chars').innerHTML = st;
}}

// ── graph experiments ─────────────────────────────────────────────────────────
const gd    = R.graph_experiments || {{}};
const gcfgs = Object.keys(gd);
if (gcfgs.length > 0) {{
  const gallA = [];
  gcfgs.forEach(function(c) {{ Object.keys((gd[c]&&gd[c].agents)||{{}}).forEach(function(a) {{
    if (gallA.indexOf(a) < 0) gallA.push(a);
  }}); }});
  function mkGDs(metric) {{
    return gallA.map(function(a) {{
      return {{ label: a, data: gcfgs.map(c => (gd[c]&&gd[c].agents&&gd[c].agents[a]&&gd[c].agents[a][metric]) || null),
               borderColor: C[a]||'#7c83fd', backgroundColor: rgba(C[a]||'#7c83fd',.1),
               borderWidth: 2, pointRadius: 5, tension: .2, fill: false }};
    }});
  }}
  line('c-gph-r',  gcfgs, mkGDs('mean_reward'));
  line('c-gph-cr', gcfgs, mkGDs('mean_cr'), {{ min:0, max:1.05, ticks:{{ callback: function(v){{return (v*100).toFixed(0)+'%';}} }} }});
  let gt = '<tr><th>Agent</th>' + gcfgs.map(c => '<th colspan="2">' + c + '</th>').join('') + '</tr>';
  gt    += '<tr><th></th>' + gcfgs.map(() => '<th>Reward</th><th>CR</th>').join('') + '</tr>';
  gallA.forEach(function(a) {{
    gt += '<tr><td><b>' + a + '</b></td>';
    gcfgs.forEach(function(c) {{
      const d = gd[c] && gd[c].agents && gd[c].agents[a];
      gt += d ? '<td>' + n(d.mean_reward,1) + '</td><td>' + cr(d.mean_cr) + '</td>' : '<td>-</td><td>-</td>';
    }});
    gt += '</tr>';
  }});
  document.getElementById('t-graph').innerHTML = gt;
}}

// ── brand experiments ─────────────────────────────────────────────────────────
const bd    = R.brand_experiments || {{}};
const bcfgs = Object.keys(bd);
if (bcfgs.length > 0) {{
  const ballA = [];
  bcfgs.forEach(function(c) {{ Object.keys((bd[c]&&bd[c].agents)||{{}}).forEach(function(a) {{
    if (ballA.indexOf(a) < 0) ballA.push(a);
  }}); }});
  function mkBDs(metric) {{
    return ballA.map(function(a) {{
      return {{ label: a, data: bcfgs.map(c => (bd[c]&&bd[c].agents&&bd[c].agents[a]&&bd[c].agents[a][metric]) || null),
               borderColor: C[a]||'#7c83fd', backgroundColor: rgba(C[a]||'#7c83fd',.1),
               borderWidth: 2, pointRadius: 5, tension: .2, fill: false }};
    }});
  }}
  line('c-br-r',  bcfgs, mkBDs('mean_reward'));
  line('c-br-cr', bcfgs, mkBDs('mean_cr'), {{ min:0, max:1.05, ticks:{{ callback: function(v){{return (v*100).toFixed(0)+'%';}} }} }});
  let bt = '<tr><th>Agent</th>' + bcfgs.map(c => '<th colspan="2">' + c + '</th>').join('') + '</tr>';
  bt    += '<tr><th></th>' + bcfgs.map(() => '<th>Reward</th><th>CR</th>').join('') + '</tr>';
  ballA.forEach(function(a) {{
    bt += '<tr><td><b>' + a + '</b></td>';
    bcfgs.forEach(function(c) {{
      const d = bd[c] && bd[c].agents && bd[c].agents[a];
      bt += d ? '<td>' + n(d.mean_reward,1) + '</td><td>' + cr(d.mean_cr) + '</td>' : '<td>-</td><td>-</td>';
    }});
    bt += '</tr>';
  }});
  document.getElementById('t-brand').innerHTML = bt;
}}

// ── parametric ────────────────────────────────────────────────────────────────
const pd = R.parametric || {{}};
function paramLine(cid, param, metric, yO) {{
  const ed = pd[param] || {{}};
  const pv = Object.keys(ed).sort(function(a,b){{return +a-+b;}});
  const ag = AG.filter(function(a){{ return pv.some(function(p){{ return ed[p]&&ed[p][a]; }}); }});
  const ds = ag.map(function(a) {{
    return {{ label: a, data: pv.map(function(p){{ return (ed[p]&&ed[p][a]&&ed[p][a][metric]) || null; }}),
             borderColor: C[a]||'#7c83fd', backgroundColor: rgba(C[a]||'#7c83fd',.1),
             borderWidth: 2, pointRadius: 5, tension: .2, fill: false }};
  }});
  line(cid, pv, ds, yO||{{}});
}}
const crY = {{ min:0, max:1.05, ticks:{{ callback: function(v){{return (v*100).toFixed(0)+'%';}} }} }};
paramLine('c-dur-cr', 'duration',   'mean_cr', crY);
paramLine('c-av-cr',  'avail_prob', 'mean_cr', crY);
paramLine('c-gd-cr',  'n_goods',    'mean_cr', crY);

let paramt = '<tr><th>Parameter</th><th>Value</th>' +
  AG.filter(function(a){{return names.indexOf(a)>=0;}}).map(function(a){{return '<th>'+a+'</th>';}}).join('') + '</tr>';
['duration','avail_prob','n_goods'].forEach(function(param) {{
  const ed = pd[param] || {{}};
  Object.keys(ed).sort(function(a,b){{return +a-+b;}}).forEach(function(v) {{
    paramt += '<tr><td>' + param + '</td><td>' + v + '</td>' +
      AG.filter(function(a){{return names.indexOf(a)>=0;}})
        .map(function(a){{ const val = ed[v]&&ed[v][a]&&ed[v][a].mean_cr;
          return '<td>' + (val!=null ? (val*100).toFixed(1)+'%' : '-') + '</td>'; }}).join('') + '</tr>';
  }});
}});
document.getElementById('t-param').innerHTML = paramt;

// ── significance ──────────────────────────────────────────────────────────────
const sig = R.significance_table || {{}};
let sh = '<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohen\'s d</th><th>Better</th></tr>';
Object.entries(sig).sort(function(a,b){{return a[1].p_value-b[1].p_value;}}).forEach(function(kv) {{
  const k = kv[0]; const v = kv[1];
  const stars = v.p_value<.001?'***':v.p_value<.01?'**':v.p_value<.05?'*':'';
  const sc    = v.significant ? 'sig' : 'ns';
  sh += '<tr><td>' + k + '</td><td>' + n(v.t_stat,3) + '</td><td>' + n(v.p_value,5) + '</td>' +
        '<td><span class="' + sc + '">' + (v.significant?'yes '+stars:'no') + '</span></td>' +
        '<td>' + n(v.cohens_d,3) + '</td><td><b>' + v.better + '</b></td></tr>';
}});
document.getElementById('t-sig').innerHTML = sh;

// ── convergence ───────────────────────────────────────────────────────────────
const tabConv  = SW.tabular || {{}};
const deepConv = SW.deep    || {{}};
const ts  = Object.keys(tabConv).sort(function(a,b){{return +a-+b;}});
const ds2 = Object.keys(deepConv).sort(function(a,b){{return +a-+b;}});

function mkConvDs(data, sizes, agts) {{
  return agts.filter(function(a){{ return sizes.some(function(s){{return data[s]&&data[s][a];}});  }})
    .map(function(a) {{
      return {{ label: a, data: sizes.map(function(s){{return (data[s]&&data[s][a]&&data[s][a].mean_reward)||null;}}),
               borderColor: C[a]||'#7c83fd', backgroundColor: rgba(C[a]||'#7c83fd',.1),
               borderWidth: 2, pointRadius: 5, tension: .15, fill: false }};
    }});
}}
line('c-tab-r',  ts.map(function(s){{return parseInt(s).toLocaleString();}}),
     mkConvDs(tabConv,  ts,  ['Q-Learning','SARSA','ExpSARSA']));
line('c-deep-r', ds2.map(function(s){{return parseInt(s).toLocaleString();}}),
     mkConvDs(deepConv, ds2, ['DQN','DoubleDQN','DuelingDQN']));

const allS = [...new Set([...ts,...ds2])].sort(function(a,b){{return +a-+b;}});
let ct = '<tr><th>Agent</th>' + allS.map(function(s){{return '<th>'+parseInt(s).toLocaleString()+'</th>';}}).join('') + '</tr>';
['Q-Learning','SARSA','ExpSARSA','DQN','DoubleDQN','DuelingDQN'].forEach(function(a) {{
  ct += '<tr><td><b>' + a + '</b></td>';
  allS.forEach(function(s) {{
    const d = (tabConv[s]&&tabConv[s][a]) || (deepConv[s]&&deepConv[s][a]);
    ct += d ? '<td>' + n(d.mean_reward,0) + '</td>' : '<td>-</td>';
  }});
  ct += '</tr>';
}});
document.getElementById('t-conv').innerHTML = ct;

// ── rankings ──────────────────────────────────────────────────────────────────
const sorted = [...names].sort(function(a,b){{return bl[b].mean_reward - bl[a].mean_reward;}});
let rt = '<tr><th>Rank</th><th>Agent</th><th>Reward</th><th>+/-CI</th><th>CR</th><th>Success Rate</th><th>Stores</th><th>Category</th></tr>';
sorted.forEach(function(a, i) {{
  const d   = bl[a];
  const rc  = i===0?'r1':i===1?'r2':i===2?'r3':'';
  const medal = i===0?'🥇':i===1?'🥈':i===2?'🥉':(i+1);
  const cat = (function() {{
    const entry = Object.entries(GROUPS).find(function(kv){{return kv[1].indexOf(a)>=0;}});
    return entry ? entry[0] : '';
  }})();
  rt += '<tr><td class="' + rc + '">' + medal + '</td>' +
        '<td><b>' + a + '</b></td><td class="' + rc + '">' + n(d.mean_reward,1) + '</td>' +
        '<td>+/-' + n(d.ci95_reward,1) + '</td><td>' + cr(d.mean_cr) + '</td>' +
        '<td>' + n(d.success_rate*100,1) + '%</td>' +
        '<td>' + n(d.mean_stores,1) + '</td><td>' + cat + '</td></tr>';
}});
document.getElementById('t-rank').innerHTML = rt;
</script>
</body>
</html>"""

# ── write output ───────────────────────────────────────────────────────────────
out = os.path.abspath(OUT_PATH)
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Dashboard saved to: {out}')
print('Open via local server to avoid browser security blocks:')
print('  1. cd to the package folder')
print('  2. python -m http.server 8080')
print('  3. Open http://127.0.0.1:8080/dashboard_fixed.html')
