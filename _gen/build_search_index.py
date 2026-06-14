"""
Build search-index.json for the client-side blog search.
Extracts each post's title, description, tags, and body text (tags stripped)
so js/search.js can match queries against the full content of every post.

Tags are defined here (single source of truth); they're also shown on the
cards/post headers in the HTML. Re-run after adding or editing a post.
"""
import html, json, os, re

ROOT = os.path.join(os.path.dirname(__file__), '..')
POSTS_DIR = os.path.join(ROOT, 'posts')

TAGS = {
    'figgie-rl.html':          ['reinforcement learning', 'self-play', 'transformers', 'games'],
    'rl-explainer.html':       ['reinforcement learning', 'theory', 'tutorial'],
    'dynamic-reweighting.html':['generalization', 'computer vision', 'self-supervised'],
}


def strip(htmltext):
    # drop scripts/styles/svg, then all tags, then collapse whitespace
    t = re.sub(r'<(script|style|svg)\b[^>]*>.*?</\1>', ' ', htmltext, flags=re.S | re.I)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = html.unescape(t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def first(pattern, s, default=''):
    m = re.search(pattern, s, re.S)
    return strip(m.group(1)) if m else default


def main():
    index = []
    for fn, tags in TAGS.items():
        path = os.path.join(POSTS_DIR, fn)
        if not os.path.exists(path):
            print('skip (missing)', fn); continue
        s = open(path, encoding='utf-8').read()
        title = first(r'<h1[^>]*class="title"[^>]*>(.*?)</h1>', s) \
            or first(r'<h1[^>]*>(.*?)</h1>', s) \
            or first(r'<title>(.*?)</title>', s)
        desc = ''
        m = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', s)
        if m:
            desc = html.unescape(m.group(1))
        art = re.search(r'<article\b[^>]*>(.*?)</article>', s, re.S)
        body = strip(art.group(1)) if art else strip(s)
        index.append({
            'url': 'posts/' + fn,
            'title': title,
            'desc': desc,
            'tags': tags,
            'text': body.lower(),
        })
        print(f'indexed {fn}: title={title[:40]!r} tags={tags} ~{len(body)} chars')

    out = os.path.join(ROOT, 'search-index.json')
    json.dump(index, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print('wrote', out, f'({os.path.getsize(out)} bytes)')


if __name__ == '__main__':
    main()
