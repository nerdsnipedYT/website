"""
Generate the interactive Plotly figures for the FiggieRL blog post.
Reads /tmp/figgie_data.json (curves extracted from tensorboard + computed fair
values) and writes self-contained *.html into ../assets/, each with a small
listener that recolors the figure when the parent page toggles dark/light.
"""
import json, os
import plotly.graph_objects as go

DATA = json.load(open('/tmp/figgie_data.json'))
ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')

# palette (matches site css accents)
BLUE   = '#5b6c8f'
TEAL   = '#2e9e8f'
ORANGE = '#e0663a'
GOLD   = '#caa24a'
PURPLE = '#8a6bd0'

DARK = dict(paper='#1a1a22', plot='#1a1a22', ink='#dadae3', grid='#2a2a34', faint='#7c7c88', zero='#3a3a46')
LIGHT = dict(paper='#ffffff', plot='#ffffff', ink='#1a1a1f', grid='#e7e7ee', faint='#8a8a96', zero='#cfcfd8')


def base_layout(th, **kw):
    lay = dict(
        paper_bgcolor=th['paper'], plot_bgcolor=th['plot'],
        font=dict(family='Inter, system-ui, sans-serif', size=13, color=th['ink']),
        margin=dict(l=62, r=20, t=44, b=86),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='top', y=-0.24, x=0, font=dict(size=12)),
        hovermode='x unified',
    )
    lay.update(kw)
    return lay


def axes(fig, th, xtitle, ytitle, **yk):
    fig.update_xaxes(title_text=xtitle, gridcolor=th['grid'], zeroline=False,
                     linecolor=th['grid'], color=th['faint'], title_font=dict(size=12.5))
    fig.update_yaxes(title_text=ytitle, gridcolor=th['grid'], zerolinecolor=th['zero'],
                     zeroline=True, linecolor=th['grid'], color=th['faint'],
                     title_font=dict(size=12.5), **yk)


def write(fig, name, height=420):
    """Serialize fig for both themes and emit a theme-aware standalone html."""
    dark = fig.to_dict()
    div_id = name
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden}}</style>
<script src="https://cdn.plot.ly/plotly-3.6.0.min.js"></script></head>
<body>
<div id="{div_id}" style="width:100%;height:{height}px"></div>
<script>
const FIG = {json.dumps(dark)};
const DARK = {json.dumps(DARK)}, LIGHT = {json.dumps(LIGHT)};
function lay(th){{
  return JSON.parse(JSON.stringify(FIG.layout));
}}
function themed(th){{
  const L = lay();
  L.paper_bgcolor=th.paper; L.plot_bgcolor=th.plot;
  L.font = L.font||{{}}; L.font.color=th.ink;
  for (const ax of ['xaxis','yaxis','xaxis2','yaxis2']){{
    if(!L[ax]) continue;
    L[ax].gridcolor=th.grid; L[ax].linecolor=th.grid; L[ax].color=th.faint;
    if(L[ax].zerolinecolor) L[ax].zerolinecolor=th.zero;
    if(L[ax].title) L[ax].title.font = {{size:12.5, color:th.faint}};
  }}
  return L;
}}
let cur = (new URLSearchParams(location.search).get('theme')||'dark');
function draw(t){{
  cur=t; const th = (t==='light')?LIGHT:DARK;
  Plotly.react("{div_id}", FIG.data, themed(th), {{displayModeBar:false, responsive:true}});
}}
draw(cur);
window.addEventListener('message', e=>{{ if(e.data&&e.data.theme) draw(e.data.theme); }});
</script></body></html>"""
    open(os.path.join(ASSETS, name + '.html'), 'w').write(html)
    print('wrote', name + '.html')


# ── Fig 1: Bayesian fair value — what your hand tells you ──────────────────────
def fig_fairvalue():
    th = DARK
    d = DATA['fv']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['x'], y=d['pgoal'], name='P(this suit is the goal)',
        mode='lines+markers', line=dict(color=TEAL, width=2.8),
        marker=dict(size=7), hovertemplate='hold %{x} → P(goal)=%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=d['x'], y=[v/d['value'][0] for v in d['value']],
        name='fair value of one card (normalised)', mode='lines',
        line=dict(color=GOLD, width=2.2, dash='dot'),
        hovertemplate='hold %{x} → relative value %{y:.2f}<extra></extra>'))
    fig.update_layout(**base_layout(th, title=dict(
        text='The suit you hold most is probably <b>not</b> the goal',
        font=dict(size=15), x=0.01, xanchor='left')))
    axes(fig, th, 'cards of a suit in your opening hand', 'probability', rangemode='tozero')
    write(fig, 'figgie_fairvalue', height=420)


# ── Fig 2: the headline learning curves (CFR run2) ────────────────────────────
def fig_learning():
    th = DARK
    fig = go.Figure()
    r = DATA['cfr2_vs_rand']; t = DATA['cfr2_vs_tom']
    fig.add_trace(go.Scatter(x=r['x'], y=r['y'], name='vs Random', mode='lines',
        line=dict(color=ORANGE, width=2.8),
        hovertemplate='%{y:.1f} chips<extra>vs Random</extra>'))
    fig.add_trace(go.Scatter(x=t['x'], y=t['y'], name='vs Tom (heuristic)', mode='lines',
        line=dict(color=BLUE, width=2.8),
        hovertemplate='%{y:.1f} chips<extra>vs Tom</extra>'))
    fig.add_hline(y=0, line=dict(color=th['zero'], width=1.2, dash='dash'))
    fig.update_layout(**base_layout(th, title=dict(
        text='Deep-CFR self-play: average chips won per game vs a fixed opponent',
        font=dict(size=14.5), x=0.01, xanchor='left')))
    axes(fig, th, 'training iteration', 'mean chips / game (0 = break-even)')
    write(fig, 'figgie_learning', height=440)


# ── Fig 3: PPO vs CFR — two algorithms, two outcomes ──────────────────────────
def fig_algos():
    th = DARK
    fig = go.Figure()
    pr = DATA['ppo_vs_rand']; pt = DATA['ppo_vs_tom']
    cr = DATA['cfr2_vs_rand']; ct = DATA['cfr2_vs_tom']
    # normalise x to [0,1] training progress so the two runs overlay
    def norm(s):
        x = s['x']; lo, hi = x[0], x[-1]; sp = (hi - lo) or 1
        return [(v - lo) / sp for v in x]
    fig.add_trace(go.Scatter(x=norm(pt), y=pt['y'], name='PPO · vs Tom',
        mode='lines', line=dict(color=TEAL, width=2.8),
        hovertemplate='%{y:.1f}<extra>PPO vs Tom</extra>'))
    fig.add_trace(go.Scatter(x=norm(ct), y=ct['y'], name='Deep-CFR · vs Tom',
        mode='lines', line=dict(color=ORANGE, width=2.8),
        hovertemplate='%{y:.1f}<extra>CFR vs Tom</extra>'))
    fig.add_trace(go.Scatter(x=norm(pr), y=pr['y'], name='PPO · vs Random',
        mode='lines', line=dict(color=TEAL, width=1.6, dash='dot'), opacity=0.7,
        hovertemplate='%{y:.1f}<extra>PPO vs Random</extra>'))
    fig.add_trace(go.Scatter(x=norm(cr), y=cr['y'], name='Deep-CFR · vs Random',
        mode='lines', line=dict(color=ORANGE, width=1.6, dash='dot'), opacity=0.7,
        hovertemplate='%{y:.1f}<extra>CFR vs Random</extra>'))
    fig.add_hline(y=0, line=dict(color=th['zero'], width=1.2, dash='dash'))
    fig.update_layout(**base_layout(th, title=dict(
        text='Two algorithms, two ceilings: only PPO crosses Tom',
        font=dict(size=14.5), x=0.01, xanchor='left')))
    axes(fig, th, 'training progress (normalised)', 'mean chips / game')
    write(fig, 'figgie_algos', height=460)


# ── Fig 4: self-play health — reward band + win rate ──────────────────────────
def fig_selfplay():
    th = DARK
    m = DATA['cfr2_rew_mean']; sd = DATA['cfr2_rew_std']; w = DATA['cfr2_winrate']
    x = m['x']
    up = [a + b for a, b in zip(m['y'], sd['y'])]
    dn = [a - b for a, b in zip(m['y'], sd['y'])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=up + dn[::-1], fill='toself',
        fillcolor='rgba(224,102,58,0.13)', line=dict(width=0), name='±1 std',
        hoverinfo='skip', showlegend=True))
    fig.add_trace(go.Scatter(x=x, y=m['y'], name='mean reward', mode='lines',
        line=dict(color=ORANGE, width=2.6), yaxis='y',
        hovertemplate='%{y:.1f}<extra>mean reward</extra>'))
    fig.add_trace(go.Scatter(x=w['x'], y=w['y'], name='win rate', mode='lines',
        line=dict(color=PURPLE, width=2.4), yaxis='y2',
        hovertemplate='%{y:.3f}<extra>win rate</extra>'))
    lay = base_layout(th, title=dict(
        text='Self-play stays near zero-sum; win rate climbs above the 0.25 floor',
        font=dict(size=14), x=0.01, xanchor='left'))
    lay['yaxis2'] = dict(overlaying='y', side='right', title_text='win rate',
        range=[0, 0.5], gridcolor='rgba(0,0,0,0)', color=th['faint'],
        zeroline=False, title_font=dict(size=12.5))
    fig.update_layout(**lay)
    fig.add_hline(y=0.25, line=dict(color=PURPLE, width=1, dash='dot'), yref='y2', opacity=0.5)
    axes(fig, th, 'training iteration', 'mean reward (chips)')
    write(fig, 'figgie_selfplay', height=440)


if __name__ == '__main__':
    fig_fairvalue()
    fig_learning()
    fig_algos()
    fig_selfplay()
    print('done')
