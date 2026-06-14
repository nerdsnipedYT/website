"""Self-contained core for the dynamic-reweighting blog assets:
MNIST + CIFAR-100 loaders, mid-layer-tap ViT, RND side towers, training, and
plotly asset builders (interactive results figures + UMAP explorer HTML)."""
import os, gzip, struct, urllib.request, tarfile, pickle, io, base64, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import plotly.graph_objects as go
from PIL import Image
from sklearn.decomposition import PCA
import umap

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
DATAROOT = os.path.dirname(__file__)

# ----------------------------------------------------------------------------- data
def load_mnist():
    base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = dict(trx='train-images-idx3-ubyte.gz', trl='train-labels-idx1-ubyte.gz',
                 tex='t10k-images-idx3-ubyte.gz', tel='t10k-labels-idx1-ubyte.gz')
    d = os.path.join(DATAROOT, 'mnist_data'); os.makedirs(d, exist_ok=True)
    def fetch(fn):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            req = urllib.request.Request(base + fn, headers={'User-Agent': 'Mozilla/5.0'})
            open(p, 'wb').write(urllib.request.urlopen(req, timeout=120).read())
        return p
    def idx(fn):
        with gzip.open(fetch(fn), 'rb') as f:
            magic, = struct.unpack('>I', f.read(4)); dims = magic & 0xFF
            shape = struct.unpack('>' + 'I' * dims, f.read(4 * dims))
            return np.frombuffer(f.read(), np.uint8).reshape(shape)
    Xtr = idx(files['trx']).astype(np.float32) / 255.; Ytr = idx(files['trl']).astype(np.int64)
    Xte = idx(files['tex']).astype(np.float32) / 255.; Yte = idx(files['tel']).astype(np.int64)
    mu, sd = Xtr.mean(), Xtr.std()
    return (Xtr - mu) / sd, Ytr, (Xte - mu) / sd, Yte, dict(mu=float(mu), sd=float(sd))

def load_cifar():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    d = os.path.join(DATAROOT, 'cifar_data'); os.makedirs(d, exist_ok=True)
    tgz = os.path.join(d, 'cifar-100-python.tar.gz')
    if not os.path.exists(os.path.join(d, 'cifar-100-python', 'train')):
        if not os.path.exists(tgz):
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            open(tgz, 'wb').write(urllib.request.urlopen(req, timeout=300).read())
        with tarfile.open(tgz) as t: t.extractall(d)
    def load(split):
        with open(os.path.join(d, 'cifar-100-python', split), 'rb') as f:
            o = pickle.load(f, encoding='bytes')
        X = o[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.
        y = np.array(o[b'fine_labels'], dtype=np.int64); return X, y
    Xtr, Ytr = load('train'); Xte, Yte = load('test')
    with open(os.path.join(d, 'cifar-100-python', 'meta'), 'rb') as f:
        names = [n.decode() for n in pickle.load(f, encoding='bytes')[b'fine_label_names']]
    mean = Xtr.mean((0, 2, 3), keepdims=True); std = Xtr.std((0, 2, 3), keepdims=True)
    return (Xtr - mean) / std, Ytr, (Xte - mean) / std, Yte, dict(mean=mean, std=std, names=names)

# ----------------------------------------------------------------------------- model
class ViT(nn.Module):
    def __init__(s, patch, d_model, depth, heads, mlp, n_cls, in_ch, img):
        super().__init__()
        s.patch, s.img, s.in_ch, s.depth = patch, img, in_ch, depth
        npp = (img // patch) ** 2
        s.embed = nn.Linear(in_ch * patch * patch, d_model)
        s.cls = nn.Parameter(torch.zeros(1, 1, d_model)); s.pos = nn.Parameter(torch.zeros(1, npp + 1, d_model))
        s.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model, heads, mlp, 0., batch_first=True, activation='gelu')
                                  for _ in range(depth)])
        s.head = nn.Linear(d_model, n_cls); s.tap_block = depth // 2
        nn.init.normal_(s.cls, std=.02); nn.init.normal_(s.pos, std=.02)
    def patchify(s, x):
        B, p, c, im = x.shape[0], s.patch, s.in_ch, s.img
        x = x.view(B, c, im, im).unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(B, -1, c * p * p)
    def tokens(s, x):
        t = s.embed(s.patchify(x)); return torch.cat([s.cls.expand(t.shape[0], -1, -1), t], 1) + s.pos
    def forward(s, x, tap=False):
        t = s.tokens(x); feat = None
        for i, blk in enumerate(s.blocks):
            t = blk(t)
            if tap and (i + 1) == s.tap_block: feat = t.mean(1)
        return (s.head(t[:, 0]), feat) if tap else s.head(t[:, 0])
    @torch.no_grad()
    def layer_embeddings(s, x):
        out = {}; t = s.tokens(x); out['patch-embed'] = t[:, 1:].mean(1)
        for i, blk in enumerate(s.blocks):
            t = blk(t); out[f'block{i + 1}'] = t[:, 0]
        return out

class Tower(nn.Module):
    def __init__(s, d_in, hidden=256, d_out=128, scale=3.0):
        super().__init__(); s.scale = scale
        s.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                              nn.Linear(hidden, d_out))
    def forward(s, x): return s.net(x.reshape(x.shape[0], -1)) * s.scale
    @torch.no_grad()
    def layer_activations(s, z):
        h = z.reshape(z.shape[0], -1); out = {}; k = 0
        for layer in s.net:
            h = layer(h)
            if isinstance(layer, nn.GELU): k += 1; out[f'hidden{k}'] = h
        out['out'] = h * s.scale; return out

def count_params(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def accuracy(model, x, y, bs=2000):
    model.eval(); ok = 0
    for i in range(0, x.shape[0], bs): ok += (model(x[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item()
    model.train(); return ok / x.shape[0]

@torch.no_grad()
def per_class_acc(model, x, y, n_cls, bs=2000):
    model.eval(); pred = torch.cat([model(x[i:i+bs]).argmax(1) for i in range(0, x.shape[0], bs)]); model.train()
    return np.array([(((pred == y) & (y == c)).sum().item() / max(1, (y == c).sum().item())) for c in range(n_cls)])

# ----------------------------------------------------------------------------- training
def train_run(cfg, xtr, ytr, xte, yte, mode='baseline', class_weight=None, steps=4000,
              batch=64, lr=3e-4, seed=0, EXP=2, normalize_weight=True, log_every=200):
    torch.manual_seed(seed); np.random.seed(seed)
    main = ViT(**cfg['vit']).to(DEV); params = list(main.parameters())
    R = P = None
    if mode == 'rnd':
        R = Tower(cfg['vit']['d_model']).to(DEV); P = Tower(cfg['vit']['d_model']).to(DEV)
        for p in R.parameters(): p.requires_grad_(False)
        params += list(P.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    hist = {'step': [], 'test': [], 'train': [], 'nov_maj': [], 'nov_min': [], 'nov_step': []}
    MAJ = cfg['majority']
    for step in range(steps + 1):
        idx = torch.randint(0, xtr.shape[0], (batch,), device=DEV); xb, yb = xtr[idx], ytr[idx]
        logits, feat = main(xb, tap=True)
        L = F.cross_entropy(logits, yb, reduction='none')
        if mode == 'rnd':
            z = feat.detach(); r = R(z); p = P(z)
            nov = ((p - r) ** EXP).mean(1); p_loss = nov.mean()
            mm = (yb == MAJ)
            hist['nov_step'].append(float(p_loss))
            hist['nov_maj'].append(float(nov[mm].mean()) if mm.any() else np.nan)
            hist['nov_min'].append(float(nov[~mm].mean()) if (~mm).any() else np.nan)
            w = nov.detach()
            if normalize_weight: w = w / (w.mean() + 1e-8)
            loss = p_loss + (w * L).mean()
        elif mode == 'normal_weight':
            w = class_weight[idx]; w = w / (w.mean() + 1e-8); loss = (w * L).mean()
        else:
            loss = L.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            hist['step'].append(step); hist['test'].append(accuracy(main, xte, yte)); hist['train'].append(accuracy(main, xtr, ytr))
    return main, R, P, hist

# ----------------------------------------------------------------------------- imbalance
def make_imbalanced(Xtr, Ytr, Xte, Yte, majority, n_train, frac_maj, n_cls, n_per_test, seed=0):
    rng = np.random.default_rng(seed)
    others = [c for c in range(n_cls) if c != majority]
    n_maj = int(round(frac_maj * n_train)); n_each = (n_train - n_maj) // len(others)
    def take(pool, k): return rng.choice(pool, k, replace=(k > len(pool)))
    parts = [take(np.where(Ytr == majority)[0], n_maj)] + [take(np.where(Ytr == c)[0], n_each) for c in others]
    sel = np.concatenate(parts); rng.shuffle(sel)
    te = np.concatenate([np.where(Yte == c)[0][:n_per_test] for c in range(n_cls)])
    xtr = torch.tensor(Xtr[sel]).to(DEV); ytr = torch.tensor(Ytr[sel]).to(DEV)
    xte = torch.tensor(Xte[te]).to(DEV); yte = torch.tensor(Yte[te]).to(DEV)
    cw = torch.where(ytr == majority, torch.full_like(ytr, n_each / n_maj, dtype=torch.float),
                     torch.ones_like(ytr, dtype=torch.float))
    return sel, te, xtr, ytr, xte, yte, cw, dict(n_maj=n_maj, n_each=n_each)

# ----------------------------------------------------------------------------- plotly helpers
PAL = dict(baseline='#5b6c8f', normal_weight='#2e9e8f', rnd='#e0663a')
def _save(fig, name, h=420):
    fig.update_layout(template='plotly_white', height=h, margin=dict(l=55, r=20, t=42, b=45),
                      font=dict(family='Inter,system-ui,sans-serif', size=13),
                      legend=dict(orientation='h', y=1.12, x=0))
    fig.write_html(os.path.join(ASSETS, name), include_plotlyjs='cdn', full_html=True,
                   config={'displayModeBar': False, 'responsive': True})

def fig_curves(hists, title, tag):
    fig = go.Figure()
    for name, h in hists.items():
        fig.add_scatter(x=h['step'], y=h['test'], mode='lines', name=name,
                        line=dict(width=2.6, color=PAL[name.split()[0] if name.split()[0] in PAL else 'baseline']))
    fig.update_layout(title=title, xaxis_title='training step', yaxis_title='macro test accuracy')
    _save(fig, f'{tag}_curves.html')

def fig_novelty(h_rnd, title, tag):
    def smooth(a, k=25):
        a = np.asarray(a, float); a = np.where(np.isnan(a), np.nanmean(a), a)
        return np.convolve(a, np.ones(k) / k, mode='valid')
    x = np.arange(24, len(h_rnd['nov_maj']))
    fig = go.Figure()
    fig.add_scatter(x=x, y=smooth(h_rnd['nov_maj']), name='majority class', line=dict(color=PAL['rnd'], width=2.6))
    fig.add_scatter(x=x, y=smooth(h_rnd['nov_min']), name='rare classes', line=dict(color=PAL['normal_weight'], width=2.6))
    fig.update_layout(title=title, xaxis_title='training step', yaxis_title='mean (P−R)²', yaxis_type='log')
    _save(fig, f'{tag}_novelty.html')

def fig_perclass_bars(pcs, labels, title, tag):  # MNIST: grouped bars
    fig = go.Figure()
    for name, pc in pcs.items():
        fig.add_bar(x=labels, y=pc, name=name, marker_color=PAL[name.split()[0] if name.split()[0] in PAL else 'baseline'])
    fig.update_layout(title=title, barmode='group', xaxis_title='digit class', yaxis_title='balanced-test accuracy')
    _save(fig, f'{tag}_perclass.html')

def fig_perclass_scatter(pc_base, pc_nw, pc_rnd, majority, names, title, tag):  # CIFAR
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#bbb'), showlegend=False)
    for pc, name, col in [(pc_nw, 'normal_weight', PAL['normal_weight']), (pc_rnd, 'novelty-RND', PAL['rnd'])]:
        fig.add_scatter(x=pc_base, y=pc, mode='markers', name=name,
                        marker=dict(color=col, size=7, opacity=.6),
                        text=names, hovertemplate='%{text}<br>base %{x:.2f} → %{y:.2f}<extra></extra>')
    fig.update_layout(title=title, xaxis_title='baseline per-class accuracy', yaxis_title='method per-class accuracy')
    _save(fig, f'{tag}_perclass.html', h=480)

# ----------------------------------------------------------------------------- UMAP explorer
def _b64(im, up=4):
    a = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    pim = Image.fromarray(a, 'L') if a.ndim == 2 else Image.fromarray(a, 'RGB')
    pim = pim.resize((a.shape[1] * up, a.shape[0] * up), Image.NEAREST)
    buf = io.BytesIO(); pim.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def _project(Z, nn_=15, md=0.0, metric='cosine', pca=50):
    Z = np.asarray(Z, np.float32); Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True); s[s < 1e-6] = 1; Z = Z / s
    if pca and Z.shape[1] > pca: Z = PCA(pca, random_state=0).fit_transform(Z)
    return umap.UMAP(n_components=2, n_neighbors=nn_, min_dist=md, metric=metric, random_state=0).fit_transform(Z)

def _img_features(imgs):
    N = len(imgs); feats = []
    if imgs.ndim == 4:
        R, G, B = imgs[..., 0], imgs[..., 1], imgs[..., 2]; lum = .299 * R + .587 * G + .114 * B
        mx, mn = imgs.max(-1), imgs.min(-1); sat = np.where(mx > 1e-6, (mx - mn) / np.clip(mx, 1e-6, None), 0.)
        m = lambda a: a.reshape(N, -1).mean(1); gy, gx = np.gradient(lum, axis=(1, 2))
        edge = np.sqrt(gx**2 + gy**2).reshape(N, -1).mean(1)
        avg = imgs.reshape(N, -1, 3).mean(1); rgbs = ['rgb(%d,%d,%d)' % tuple((c * 255).clip(0, 255).astype(int)) for c in avg]
        feats += [('brightness', m(lum), 'Cividis', False), ('saturation', m(sat), 'Viridis', False),
                  ('avg red', m(R), 'Reds', False), ('avg green', m(G), 'Greens', False), ('avg blue', m(B), 'Blues', False),
                  ('warmth (R−B)', m(R) - m(B), 'RdBu', False), ('edge density', edge, 'Magma', False),
                  ('avg color', rgbs, None, True)]
    else:
        I = imgs.reshape(N, -1); gy, gx = np.gradient(imgs, axis=(1, 2)); edge = np.sqrt(gx**2 + gy**2).reshape(N, -1).mean(1)
        feats += [('brightness', I.mean(1), 'Cividis', False), ('ink amount', (imgs > .5).reshape(N, -1).mean(1), 'Viridis', False),
                  ('contrast', I.std(1), 'Cividis', False), ('edge density', edge, 'Magma', False)]
    return feats

def build_umap(model, P, R, x_vis, y_vis, imgs_vis, label_fn, majority, out_path, title, height=560):
    emb = dict(model.layer_embeddings(x_vis))
    with torch.no_grad(): _, feat = model(x_vis, tap=True)
    for tag, tw in [('P', P), ('R', R)]:
        for nm, act in tw.layer_activations(feat).items(): emb[f'{tag}:{nm}'] = act
    names = list(emb.keys())
    coords = {n: _project(emb[n].detach().cpu().numpy()) for n in names}
    IMGS = [_b64(imgs_vis[i]) for i in range(len(y_vis))]
    LBLS = [label_fn(int(t)) for t in y_vis]
    color_opts = [('label', y_vis, 'Turbo', False)] + _img_features(imgs_vis)
    def restyle(nm, vals, scale, rgb):
        if rgb: return {'marker.color': [list(vals)], 'marker.showscale': [False]}
        return {'marker.color': [np.asarray(vals, float).tolist()], 'marker.colorscale': [scale],
                'marker.showscale': [True], 'marker.colorbar.title.text': [nm]}
    cbtn = [dict(label=nm, method='restyle', args=[restyle(nm, v, s, rgb)]) for (nm, v, s, rgb) in color_opts]
    is_maj = (y_vis == majority); ntr = len(names)
    fig = go.Figure()
    for li, n in enumerate(names):
        c = coords[n]
        fig.add_scattergl(x=c[:, 0], y=c[:, 1], mode='markers', name=n, visible=(li == 0),
                          marker=dict(size=5, color=y_vis, colorscale='Turbo', showscale=True, colorbar=dict(title='label'),
                                      line=dict(width=[1.3 if m else 0 for m in is_maj], color='black')),
                          hovertext=LBLS, hovertemplate='%{hovertext}<extra></extra>')
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=8, r=8, t=70, b=8),
                      font=dict(family='Inter,system-ui,sans-serif', size=12),
                      updatemenus=[dict(active=0, x=0, y=1.13, xanchor='left', buttons=[
                          dict(label=n, method='update', args=[{'visible': [j == i for j in range(ntr)]}]) for i, n in enumerate(names)]),
                          dict(active=0, x=0.4, y=1.13, xanchor='left', buttons=cbtn)],
                      annotations=[dict(text='layer', x=0, y=1.18, xref='paper', yref='paper', showarrow=False, font=dict(size=11, color='#888')),
                                   dict(text='color by', x=0.4, y=1.18, xref='paper', yref='paper', showarrow=False, font=dict(size=11, color='#888'))])
    plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='umap-div')
    CSS = ("<style>body{margin:0}.uw{display:flex;align-items:flex-start;gap:16px;font-family:Inter,system-ui,sans-serif}"
           ".us{position:sticky;top:8px;width:170px;text-align:center;padding:12px;border:1px solid #e3e3ea;border-radius:14px;"
           "background:#fafafe;box-shadow:0 2px 10px rgba(30,40,90,.07)}.us h4{margin:0 0 8px;font-size:10px;letter-spacing:.06em;"
           "text-transform:uppercase;color:#9a9aa8}#ul{font-weight:700;color:#1f2d3d;min-height:1.2em;margin-bottom:8px;font-size:14px}"
           "#ui{width:140px;height:140px;image-rendering:pixelated;border-radius:9px;border:1px solid #d7d7e0;background:#fff}"
           ".uh{margin-top:8px;font-size:10px;color:#9a9aa6}</style>")
    SIDE = ('<div class="us"><h4>hovered point</h4><div id="ul">&mdash;</div>'
            '<img id="ui" alt=""/><div class="uh">hover a point to preview its image</div></div>')
    JS = ("<script>var IMGS=__I__;var LBLS=__L__;(function a(){var g=document.getElementById('umap-div');"
          "if(g&&g.on){g.on('plotly_hover',function(d){var i=d.points[0].pointNumber;"
          "document.getElementById('ui').src=IMGS[i];document.getElementById('ul').innerText=LBLS[i];});}"
          "else{setTimeout(a,300);}})();</script>")
    JS = JS.replace('__I__', json.dumps(IMGS)).replace('__L__', json.dumps([str(x) for x in LBLS]))
    html = ('<!doctype html><html><head><meta charset="utf-8"><title>' + title + '</title>' + CSS +
            '</head><body><div class="uw">' + plot_div + SIDE + '</div>' + JS + '</body></html>')
    open(out_path, 'w').write(html)
    return names
