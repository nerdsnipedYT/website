import sys, json, os, time, numpy as np, torch
import core as C

def run_mnist():
    t0 = time.time()
    Xtr, Ytr, Xte, Yte, meta = C.load_mnist()
    cfg = dict(vit=dict(patch=7, d_model=64, depth=4, heads=4, mlp=128, n_cls=10, in_ch=1, img=28),
               majority=8, n_cls=10)
    sel, te, xtr, ytr, xte, yte, cw, cnt = C.make_imbalanced(Xtr, Ytr, Xte, Yte, 8, 10000, 0.91, 10, 200)
    print('MNIST pool', xtr.shape, 'test', xte.shape, cnt)
    base, _, _, hb = C.train_run(cfg, xtr, ytr, xte, yte, 'baseline', steps=4000, seed=0)
    nw, _, _, hn = C.train_run(cfg, xtr, ytr, xte, yte, 'normal_weight', class_weight=cw, steps=4000, seed=0)
    rnd, R, P, hr = C.train_run(cfg, xtr, ytr, xte, yte, 'rnd', steps=4000, seed=0)
    pcs = {'baseline': C.per_class_acc(base, xte, yte, 10),
           'normal_weight': C.per_class_acc(nw, xte, yte, 10),
           'novelty-RND': C.per_class_acc(rnd, xte, yte, 10)}
    C.fig_curves({'baseline': hb, 'normal_weight': hn, 'novelty-RND': hr},
                 'Imbalanced MNIST — macro test accuracy (majority = digit 8, 91%)', 'mnist')
    C.fig_novelty(hr, 'Novelty (P−R)² over training — majority vs rare', 'mnist')
    C.fig_perclass_bars(pcs, [str(i) for i in range(10)], 'Per-class balanced-test accuracy', 'mnist')
    # umap on balanced test
    rng = np.random.default_rng(0); vs = rng.choice(len(te), size=1500, replace=False)
    x_vis = xte[vs]; y_vis = yte[vs].cpu().numpy()
    imgs = (Xte[te][vs] * meta['sd'] + meta['mu']).clip(0, 1)
    layers = C.build_umap(rnd, P, R, x_vis, y_vis, imgs, lambda c: str(c), 8,
                          os.path.join(C.ASSETS, 'mnist_umap.html'),
                          'Imbalanced MNIST — layer UMAP explorer')
    # multi-seed best macro
    rows = []
    for s in [0, 1, 2]:
        _, _, _, a = C.train_run(cfg, xtr, ytr, xte, yte, 'baseline', steps=3000, seed=s, log_every=200)
        _, _, _, b = C.train_run(cfg, xtr, ytr, xte, yte, 'normal_weight', class_weight=cw, steps=3000, seed=s)
        _, _, _, d = C.train_run(cfg, xtr, ytr, xte, yte, 'rnd', steps=3000, seed=s)
        rows.append([max(a['test']), max(b['test']), max(d['test'])])
    rows = np.array(rows)
    res = dict(
        params_vit=C.count_params(base), params_tower=C.count_params(P),
        n_maj=cnt['n_maj'], n_each=cnt['n_each'], tap_block=base.tap_block, depth=base.depth,
        best=dict(baseline=float(max(hb['test'])), normal_weight=float(max(hn['test'])), rnd=float(max(hr['test']))),
        final=dict(baseline=float(hb['test'][-1]), normal_weight=float(hn['test'][-1]), rnd=float(hr['test'][-1])),
        train_final=dict(baseline=float(hb['train'][-1]), normal_weight=float(hn['train'][-1]), rnd=float(hr['train'][-1])),
        majority_acc=dict(baseline=float(pcs['baseline'][8]), normal_weight=float(pcs['normal_weight'][8]), rnd=float(pcs['novelty-RND'][8])),
        rare_acc=dict(baseline=float(np.delete(pcs['baseline'], 8).mean()),
                      normal_weight=float(np.delete(pcs['normal_weight'], 8).mean()),
                      rnd=float(np.delete(pcs['novelty-RND'], 8).mean())),
        nov_ratio=float(np.nanmean(hr['nov_min'][-200:]) / max(np.nanmean(hr['nov_maj'][-200:]), 1e-9)),
        seeds=dict(baseline=rows[:, 0].tolist(), normal_weight=rows[:, 1].tolist(), rnd=rows[:, 2].tolist()),
        seeds_mean=dict(baseline=float(rows[:, 0].mean()), normal_weight=float(rows[:, 1].mean()), rnd=float(rows[:, 2].mean())),
        umap_layers=layers)
    json.dump(res, open(os.path.join(C.ASSETS, 'mnist_results.json'), 'w'), indent=2)
    print('MNIST done in %.0fs' % (time.time() - t0)); print(json.dumps(res['best']), json.dumps(res['seeds_mean']))

def run_cifar():
    t0 = time.time()
    Xtr, Ytr, Xte, Yte, meta = C.load_cifar()
    names = meta['names']; MAJ = names.index('sunflower')
    cfg = dict(vit=dict(patch=8, d_model=128, depth=6, heads=4, mlp=256, n_cls=100, in_ch=3, img=32),
               majority=MAJ, n_cls=100)
    sel, te, xtr, ytr, xte, yte, cw, cnt = C.make_imbalanced(Xtr, Ytr, Xte, Yte, MAJ, 20000, 0.90, 100, 50)
    print('CIFAR pool', xtr.shape, 'test', xte.shape, 'majority', names[MAJ], cnt)
    base, _, _, hb = C.train_run(cfg, xtr, ytr, xte, yte, 'baseline', steps=4000, seed=0)
    nw, _, _, hn = C.train_run(cfg, xtr, ytr, xte, yte, 'normal_weight', class_weight=cw, steps=4000, seed=0)
    rnd, R, P, hr = C.train_run(cfg, xtr, ytr, xte, yte, 'rnd', steps=4000, seed=0)
    pcb = C.per_class_acc(base, xte, yte, 100); pcn = C.per_class_acc(nw, xte, yte, 100); pcr = C.per_class_acc(rnd, xte, yte, 100)
    C.fig_curves({'baseline': hb, 'normal_weight': hn, 'novelty-RND': hr},
                 'Imbalanced CIFAR-100 — macro test accuracy (majority = "%s", 90%%)' % names[MAJ], 'cifar')
    C.fig_novelty(hr, 'Novelty (P−R)² over training — majority vs rare', 'cifar')
    C.fig_perclass_scatter(pcb, pcn, pcr, MAJ, names, 'Per-class accuracy: method vs baseline', 'cifar')
    def destd(arr): return np.clip(np.moveaxis(arr * meta['std'][0] + meta['mean'][0], -3, -1), 0, 1)
    rng = np.random.default_rng(0); vs = rng.choice(len(te), size=1800, replace=False)
    x_vis = xte[vs]; y_vis = yte[vs].cpu().numpy(); imgs = destd(Xte[te][vs])
    layers = C.build_umap(rnd, P, R, x_vis, y_vis, imgs, lambda c: names[c], MAJ,
                          os.path.join(C.ASSETS, 'cifar_umap.html'), 'Imbalanced CIFAR-100 — layer UMAP explorer')
    rare = [c for c in range(100) if c != MAJ]
    res = dict(majority=names[MAJ], params_vit=C.count_params(base), params_tower=C.count_params(P),
               n_maj=cnt['n_maj'], n_each=cnt['n_each'], tap_block=base.tap_block, depth=base.depth,
               best=dict(baseline=float(max(hb['test'])), normal_weight=float(max(hn['test'])), rnd=float(max(hr['test']))),
               train_final=dict(baseline=float(hb['train'][-1]), normal_weight=float(hn['train'][-1]), rnd=float(hr['train'][-1])),
               majority_acc=dict(baseline=float(pcb[MAJ]), normal_weight=float(pcn[MAJ]), rnd=float(pcr[MAJ])),
               rare_acc=dict(baseline=float(pcb[rare].mean()), normal_weight=float(pcn[rare].mean()), rnd=float(pcr[rare].mean())),
               nov_ratio=float(np.nanmean(hr['nov_min'][-200:]) / max(np.nanmean(hr['nov_maj'][-200:]), 1e-9)),
               umap_layers=layers)
    json.dump(res, open(os.path.join(C.ASSETS, 'cifar_results.json'), 'w'), indent=2)
    print('CIFAR done in %.0fs' % (time.time() - t0)); print(json.dumps(res['best']))

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'mnist'
    if which == 'mnist': run_mnist()
    else: run_cifar()
