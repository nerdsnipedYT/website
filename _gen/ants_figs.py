"""Two theme-neutral SVG diagrams for the 'ants on a stick' puzzle post."""
import os

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
WARM = '#e8714e'   # ants facing right
COOL = '#4f86c6'   # ants facing left
GRAY = '#9a9aa6'
STICK = '#b79b74'
WATER = '#7fb6c4'
DOT = '#caa24a'


def ant(x, y, color, facing):
    """A little ant: 3 body blobs + legs + a direction arrow above."""
    d = 1 if facing == 'right' else -1
    g = []
    # legs
    for dx in (-3, 0, 3):
        g.append(f'<line x1="{x+dx}" y1="{y}" x2="{x+dx-2*d}" y2="{y+4}" stroke="#3a3a42" stroke-width="0.8"/>')
        g.append(f'<line x1="{x+dx}" y1="{y}" x2="{x+dx-2*d}" y2="{y-4}" stroke="#3a3a42" stroke-width="0.8"/>')
    # body (head leads in facing direction)
    g.append(f'<circle cx="{x+5*d}" cy="{y}" r="3.2" fill="{color}"/>')
    g.append(f'<circle cx="{x}" cy="{y}" r="3.6" fill="{color}"/>')
    g.append(f'<circle cx="{x-5*d}" cy="{y}" r="3" fill="{color}"/>')
    # direction arrow above
    ax = x + d * 9
    g.append(f'<line x1="{x-d*9}" y1="{y-12}" x2="{ax}" y2="{y-12}" stroke="{color}" stroke-width="1.6"/>')
    g.append(f'<path d="M{ax},{y-12} L{ax-d*4},{y-14.5} L{ax-d*4},{y-9.5} Z" fill="{color}"/>')
    return ''.join(g)


def setup():
    W, H = 480, 150
    xs = [40 + i * 44 for i in range(10)]          # 10 ant positions
    g = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, system-ui, sans-serif">']
    # water
    g.append(f'<rect x="0" y="112" width="{W}" height="38" fill="{WATER}" opacity="0.18"/>')
    # stick
    g.append(f'<rect x="18" y="96" width="{W-36}" height="16" rx="3" fill="{STICK}" stroke="#8f7651" stroke-width="1"/>')
    for i, x in enumerate(xs):
        facing = 'right' if i < 5 else 'left'
        color = WARM if i < 5 else COOL
        g.append(ant(x, 88, color, facing))
    g.append(f'<text x="{W/2}" y="140" text-anchor="middle" font-size="11" fill="{GRAY}">'
             f'5 ants face right · 5 face left — count the collisions before all drown</text>')
    g.append('</svg>')
    open(os.path.join(ASSETS, 'ants_setup.svg'), 'w').write(''.join(g))
    print('wrote ants_setup.svg')


def spacetime():
    """Position (x) horizontal, time (t) upward. Ghosts = straight world-lines;
    every left-mover crosses every right-mover exactly once -> 5x5 = 25 dots."""
    W, H = 480, 330
    x0, x1 = 30, 450
    yb, yt = 286, 40                      # t=0 at bottom, t=max at top
    left_start = [70, 110, 150, 190, 230]   # face right -> drift +x with time
    right_start = [250, 290, 330, 370, 410] # face left  -> drift -x with time
    drift = 182                              # horizontal travel over full height
    g = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, system-ui, sans-serif">']
    # axes
    g.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{GRAY}" stroke-width="1.2" opacity=".6"/>')
    g.append(f'<line x1="{x0}" y1="{yb}" x2="{x0}" y2="{yt-4}" stroke="{GRAY}" stroke-width="1.2" opacity=".6"/>')
    g.append(f'<text x="{x1}" y="{yb+16}" text-anchor="end" font-size="10" fill="{GRAY}">position along the stick →</text>')
    g.append(f'<text x="{x0-6}" y="{yt+6}" text-anchor="end" font-size="10" fill="{GRAY}" transform="rotate(-90 {x0-6} {yt+6})">time →</text>')

    def line_at(start, sign):
        # returns endpoints (bottom -> top)
        return (start, yb), (start + sign * drift, yt)

    lefts = [line_at(s, +1) for s in left_start]
    rights = [line_at(s, -1) for s in right_start]
    for (bx, by), (tx, ty) in lefts:
        g.append(f'<line x1="{bx}" y1="{by}" x2="{tx}" y2="{ty}" stroke="{WARM}" stroke-width="2.4" stroke-linecap="round" opacity=".9"/>')
    for (bx, by), (tx, ty) in rights:
        g.append(f'<line x1="{bx}" y1="{by}" x2="{tx}" y2="{ty}" stroke="{COOL}" stroke-width="2.4" stroke-linecap="round" opacity=".9"/>')

    # intersections of every left with every right
    def inter(a, b):
        (x1a, y1a), (x2a, y2a) = a
        (x1b, y1b), (x2b, y2b) = b
        d = (x1a - x2a) * (y1b - y2b) - (y1a - y2a) * (x1b - x2b)
        if d == 0:
            return None
        px = ((x1a * y2a - y1a * x2a) * (x1b - x2b) - (x1a - x2a) * (x1b * y2b - y1b * x2b)) / d
        py = ((x1a * y2a - y1a * x2a) * (y1b - y2b) - (y1a - y2a) * (x1b * y2b - y1b * x2b)) / d
        return px, py

    n = 0
    for a in lefts:
        for b in rights:
            p = inter(a, b)
            if p and x0 <= p[0] <= x1 and yt <= p[1] <= yb:
                g.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.4" fill="{DOT}" stroke="#2a2a32" stroke-width="0.8"/>')
                n += 1
    g.append(f'<text x="{W/2}" y="{yt-14}" text-anchor="middle" font-size="13" fill="{DOT}" font-weight="700">'
             f'{n} crossings = {n} collisions</text>')
    g.append('</svg>')
    open(os.path.join(ASSETS, 'ants_spacetime.svg'), 'w').write(''.join(g))
    print(f'wrote ants_spacetime.svg ({n} crossings)')


if __name__ == '__main__':
    setup()
    spacetime()
    print('done')
