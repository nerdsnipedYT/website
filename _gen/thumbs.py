"""
Generate small, theme-neutral SVG thumbnails for the index post cards.
Colors are chosen to read on both the dark (#1a1a22) and light (#ffffff)
card surfaces — accent hues + a mid-gray for axes, transparent background.
Each is 200×150 (4:3).
"""
import math, os, random

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
ORANGE = '#e8714e'; TEAL = '#37a08f'; BLUE = '#6b80a8'; PURPLE = '#8a6bd0'
GRAY = '#9a9aa6'; RED = '#d4634a'

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150" '
        'font-family="Inter, system-ui, sans-serif">')


def save(name, body):
    open(os.path.join(ASSETS, name), 'w').write(HEAD + body + '</svg>\n')
    print('wrote', name)


def smooth(pts):
    """Catmull-Rom-ish smooth path through points."""
    d = f'M{pts[0][0]:.1f},{pts[0][1]:.1f}'
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, len(pts) - 1)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += f' C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}'
    return d


# ── 1) Figgie build log: learning curves crossing the break-even line ──────────
def figgie():
    b = []
    # axes
    b.append(f'<line x1="26" y1="20" x2="26" y2="128" stroke="{GRAY}" stroke-width="1.4" opacity=".55"/>')
    b.append(f'<line x1="26" y1="128" x2="188" y2="128" stroke="{GRAY}" stroke-width="1.4" opacity=".55"/>')
    # break-even (zero) dashed line
    b.append(f'<line x1="26" y1="82" x2="188" y2="82" stroke="{GRAY}" stroke-width="1.2" '
             f'stroke-dasharray="4 4" opacity=".7"/>')
    b.append(f'<text x="30" y="79" font-size="8" fill="{GRAY}">0</text>')
    # vs Random — climbs high
    rnd = [(26, 120), (66, 96), (104, 70), (146, 46), (186, 32)]
    b.append(f'<path d="{smooth(rnd)}" fill="none" stroke="{ORANGE}" stroke-width="3" '
             f'stroke-linecap="round"/>')
    # vs Tom — starts below zero, crosses up
    tom = [(26, 112), (66, 100), (104, 86), (146, 72), (186, 58)]
    b.append(f'<path d="{smooth(tom)}" fill="none" stroke="{BLUE}" stroke-width="3" '
             f'stroke-linecap="round"/>')
    # crossing marker
    b.append(f'<circle cx="128" cy="82" r="3.4" fill="{BLUE}"/>')
    # suit motif
    b.append(f'<text x="150" y="20" font-size="15" fill="{RED}">♦</text>')
    b.append(f'<text x="168" y="20" font-size="15" fill="{GRAY}">♠</text>')
    save('thumb_figgie.svg', ''.join(b))


# ── 2) RL explainer: gradient ascent toward the optimum on a loss landscape ────
def rl():
    b = []
    cx, cy = 138, 74
    for rx, ry, op in [(54, 40, .35), (38, 28, .5), (22, 16, .7)]:
        b.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" '
                 f'stroke="{GRAY}" stroke-width="1.3" opacity="{op}"/>')
    b.append(f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="{PURPLE}"/>')
    # ascent path with arrowhead
    path = [(30, 132), (58, 116), (84, 104), (108, 90), (128, 80)]
    b.append(f'<path d="{smooth(path)}" fill="none" stroke="{ORANGE}" stroke-width="3" '
             f'stroke-linecap="round"/>')
    # arrowhead pointing at optimum
    ang = math.atan2(80 - 90, 128 - 108)
    ax, ay = 130, 79
    for s in (1, -1):
        a2 = ang + s * 2.6
        b.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ax-8*math.cos(a2):.1f}" '
                 f'y2="{ay-8*math.sin(a2):.1f}" stroke="{ORANGE}" stroke-width="3" stroke-linecap="round"/>')
    b.append(f'<text x="24" y="128" font-size="15" fill="{ORANGE}" font-style="italic">∇J</text>')
    b.append(f'<text x="150" y="60" font-size="13" fill="{PURPLE}">π*</text>')
    save('thumb_rl.svg', ''.join(b))


# ── 3) Dynamic reweighting: UMAP — a dense memorized island + sparse rare points ─
def reweight():
    rng = random.Random(7)
    b = []
    # dense, over-memorized majority island (low novelty) — tight cluster
    cxx, cyy = 70, 92
    dense = []
    for _ in range(16):
        a = rng.uniform(0, 2 * math.pi); r = rng.uniform(0, 17)
        dense.append((cxx + r * math.cos(a), cyy + r * math.sin(a) * 0.8))
    # dashed ring around the island
    b.append(f'<ellipse cx="{cxx}" cy="{cyy}" rx="24" ry="20" fill="none" stroke="{GRAY}" '
             f'stroke-width="1.2" stroke-dasharray="4 4" opacity=".7"/>')
    for x, y in dense:
        b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{ORANGE}" '
                 f'stroke="#2a2a32" stroke-width="1" opacity=".95"/>')
    # sparse rare classes scattered around
    for _ in range(15):
        x = rng.uniform(110, 188); y = rng.uniform(20, 130)
        col = rng.choice([TEAL, BLUE, PURPLE])
        b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{col}" opacity=".9"/>')
    # a few stragglers upper-left
    for _ in range(4):
        x = rng.uniform(26, 60); y = rng.uniform(20, 50)
        b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{TEAL}" opacity=".9"/>')
    save('thumb_reweight.svg', ''.join(b))


if __name__ == '__main__':
    figgie(); rl(); reweight()
    print('done')
