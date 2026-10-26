"""Probe the word2vec encoding space: does it separate turn_left from turn_right, and what do
representative commands look like once encoded?

Usage:
  PYTHONPATH=scripts python3 -m doplan_lang.embedding_space \
      --csv nuplan_data/doplan/release_annotationsV1.csv \
      --kv outputs/doplan_lang/word2vec.kv --clf outputs/doplan_lang/intent_clf.joblib \
      --out-dir outputs/doplan_lang/embedding_space
Writes embedding_space.md plus fig1_left_right.png, fig2_command_space.png, fig3_tsne_by_intent.png,
fig4_command_encoding.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EVAL_LOGS, INTENTS
from .intent import dataset, predict_proba, sentence_vector
from .text import load_annotations, rule_intent, swap_left_right, tokenize

# Representative passenger commands, two or three per intent, plus a minimal left/right pair.
COMMANDS = (
    'turn left at the intersection',
    'turn right at the intersection',
    'make a left turn at the light',
    'take the next right',
    'make a u turn',
    'change to the left lane',
    'merge into the right lane',
    'stop at the stop sign',
    'pull over here',
    'go straight through the intersection',
    'keep going straight',
    'stay in the right lane',
    'head toward the hotel',
    'go around the roundabout',
    'turn right from the left lane',
)

# Validated categorical palette (dataviz skill), fixed slot per intent.
COLORS = {'turn_left': '#2a78d6', 'turn_right': '#eb6834', 'lane_left': '#1baf7a', 'lane_right': '#eda100',
          'stop': '#e87ba4', 'straight': '#008300', 'other': '#6250d6'}
MARKERS = {'turn_left': 'o', 'turn_right': 's', 'lane_left': '^', 'lane_right': 'v',
           'stop': 'D', 'straight': 'P', 'other': 'X'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
DIVERGING = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#f4f8fd', '#cde2fb', '#5598e7', '#1c5cab', '#0d366b'])

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


def row_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return np.sum(A * B, axis=1)


def word_level(kv: KeyedVectors) -> dict:
    left, right = kv['left'], kv['right']
    nn_left = [w for w, _ in kv.most_similar('left', topn=len(kv.key_to_index) - 1)]
    nn_right = [w for w, _ in kv.most_similar('right', topn=len(kv.key_to_index) - 1)]
    return {'cos_left_right': cos(left, right),
            'rank_of_right_among_left_neighbours': nn_left.index('right') + 1,
            'rank_of_left_among_right_neighbours': nn_right.index('left') + 1,
            'norm_left': float(np.linalg.norm(left)), 'norm_right': float(np.linalg.norm(right)),
            'norm_difference': float(np.linalg.norm(left - right))}


def axis_stats(X: np.ndarray, y: np.ndarray, d: np.ndarray, pos: str, neg: str) -> dict:
    m = (y == pos) | (y == neg)
    p = X[m] @ d
    return {'n_pos': int((y == pos).sum()), 'n_neg': int((y == neg).sum()),
            'auc': float(roc_auc_score((y[m] == pos).astype(int), p)),
            'mean_pos': float(p[y[m] == pos].mean()), 'mean_neg': float(p[y[m] == neg].mean()),
            'std_pooled': float(np.sqrt(0.5 * (p[y[m] == pos].var() + p[y[m] == neg].var())))}


def clause_level(X: np.ndarray, y: np.ndarray, kv: KeyedVectors, clf, rows, seed: int = 0) -> dict:
    out: dict = {}
    d = unit(kv['left'] - kv['right'])
    cent = {k: X[y == k].mean(axis=0) for k in INTENTS if (y == k).any()}
    d_c = unit(cent['turn_left'] - cent['turn_right'])
    out['cos_word_axis_vs_centroid_axis'] = cos(d, d_c)
    out['turn_on_word_axis'] = axis_stats(X, y, d, 'turn_left', 'turn_right')
    out['lane_on_word_axis'] = axis_stats(X, y, d, 'lane_left', 'lane_right')
    out['turn_on_centroid_axis'] = axis_stats(X, y, d_c, 'turn_left', 'turn_right')
    proj = X @ d
    out['projection_on_word_axis_by_intent'] = {k: {'mean': float(proj[y == k].mean()), 'sd': float(proj[y == k].std())}
                                                for k in INTENTS if (y == k).any()}
    W = kv.vectors @ d
    order = np.argsort(W)
    out['word_axis_top_words'] = {
        'left_pole': [(kv.index_to_key[i], round(float(W[i]), 2)) for i in order[::-1][:8]],
        'right_pole': [(kv.index_to_key[i], round(float(W[i]), 2)) for i in order[:8]],
        'rank_of_left_from_left_pole': int((W > W[kv.key_to_index['left']]).sum()) + 1,
        'rank_of_right_from_right_pole': int((W < W[kv.key_to_index['right']]).sum()) + 1,
        'vocabulary': int(len(W))}

    pca = PCA(n_components=20, random_state=seed).fit(X)
    out['variance_fraction_along_word_axis'] = float(np.var(X @ d) / np.var(X, axis=0).sum())
    out['pca_variance_ratio_first5'] = [round(float(v), 3) for v in pca.explained_variance_ratio_[:5]]
    align = np.abs(pca.components_ @ d)
    out['most_aligned_pc'] = int(np.argmax(align) + 1)
    out['abs_cos_word_axis_vs_pcs_1to5'] = [round(float(v), 3) for v in align[:5]]

    # Parallelogram: the same left/right axis in turn and lane contexts; the same turn/lane axis on both sides.
    out['cos_direction_axis_turn_vs_lane'] = cos(cent['turn_left'] - cent['turn_right'], cent['lane_left'] - cent['lane_right'])
    out['cos_manoeuvre_axis_left_vs_right'] = cos(cent['turn_left'] - cent['lane_left'], cent['turn_right'] - cent['lane_right'])
    out['cos_centroids_turn_left_turn_right'] = cos(cent['turn_left'], cent['turn_right'])
    out['cos_centroids_turn_left_stop'] = cos(cent['turn_left'], cent['stop'])
    out['cos_centroids_turn_left_straight'] = cos(cent['turn_left'], cent['straight'])

    m = (y == 'turn_left') | (y == 'turn_right')
    Xm, ym = X[m], y[m]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xm), size=min(2000, len(Xm)), replace=False)
    out['silhouette_cosine_left_vs_right'] = float(silhouette_score(Xm[idx], ym[idx], metric='cosine'))
    pred_nc = np.where(row_cos(Xm, np.tile(cent['turn_left'], (len(Xm), 1))) >=
                       row_cos(Xm, np.tile(cent['turn_right'], (len(Xm), 1))), 'turn_left', 'turn_right')
    out['nearest_centroid_accuracy_left_vs_right'] = float((pred_nc == ym).mean())
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    out['logreg_raw_cv_accuracy_left_vs_right'] = float(cross_val_score(
        LogisticRegression(C=1.0, max_iter=2000), Xm, ym, cv=skf).mean())
    out['logreg_standardised_cv_accuracy_left_vs_right'] = float(cross_val_score(
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), Xm, ym, cv=skf).mean())

    # Counterfactual: swap left<->right in every turn clause and see how far the vector moves and whether
    # the classifier follows.
    rows_m = [r for r in rows if r.rule_intent in ('turn_left', 'turn_right')]
    swapped_text = [swap_left_right(r.first_clause) for r in rows_m]
    Xs = np.stack([sentence_vector(tokenize(t), kv) for t in swapped_text])
    keep = np.any(Xm, axis=1) & np.any(Xs, axis=1)
    cos_swap = row_cos(Xm[keep], Xs[keep])
    same = np.array([rng.choice(np.flatnonzero(ym == lab)) for lab in ym])
    other = np.array([rng.choice(np.flatnonzero(ym != lab)) for lab in ym])
    cos_same = row_cos(Xm[keep], Xm[same][keep])
    cos_other = row_cos(Xm[keep], Xm[other][keep])
    flipped, orig_ok, kept_dir, n_tok = [], [], [], []
    for r, t in zip(rows_m, swapped_text):
        want = 'turn_right' if r.rule_intent == 'turn_left' else 'turn_left'
        if rule_intent(t) != want:
            continue   # swap changed the rule label unexpectedly (e.g. both sides present); skip
        p0 = predict_proba(clf, kv, r.first_clause)
        p = predict_proba(clf, kv, t)
        top = max(p, key=p.get)
        orig_ok.append(max(p0, key=p0.get) == r.rule_intent)
        flipped.append(top == want)
        kept_dir.append(top == r.rule_intent)
        n_tok.append(len(tokenize(r.first_clause)))
    flipped, orig_ok, kept_dir, n_tok = map(np.array, (flipped, orig_ok, kept_dir, n_tok))
    out['counterfactual'] = {
        'n': int(keep.sum()),
        'cos_original_vs_swapped': {'median': float(np.median(cos_swap)), 'p10': float(np.percentile(cos_swap, 10)),
                                    'p90': float(np.percentile(cos_swap, 90))},
        'cos_original_vs_random_same_intent': {'median': float(np.median(cos_same)),
                                               'p10': float(np.percentile(cos_same, 10)),
                                               'p90': float(np.percentile(cos_same, 90))},
        'cos_original_vs_random_opposite_intent': {'median': float(np.median(cos_other)),
                                                   'p10': float(np.percentile(cos_other, 10)),
                                                   'p90': float(np.percentile(cos_other, 90))},
        'classifier_flip_rate': float(flipped.mean()), 'n_flip_tested': int(len(flipped)),
        'flip_rate_when_original_classified_correctly': float(flipped[orig_ok].mean()),
        'rate_still_predicting_original_direction': float(kept_dir.mean()),
        'mean_tokens_flipped': float(n_tok[flipped].mean()), 'mean_tokens_not_flipped': float(n_tok[~flipped].mean())}
    out['_arrays'] = {'d': d, 'cent': cent, 'cos_swap': cos_swap, 'cos_same': cos_same, 'cos_other': cos_other,
                      'pca': pca}
    return out


def fig_left_right(X, y, stats, out: Path) -> None:
    d = stats['_arrays']['d']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    bins = np.linspace(-0.9, 0.9, 46)
    for ax, (pos, neg), key in zip(axes[:2], (('turn_left', 'turn_right'), ('lane_left', 'lane_right')),
                                   ('turn_on_word_axis', 'lane_on_word_axis')):
        for lab in (pos, neg):
            p = X[y == lab] @ d
            ax.hist(p, bins=bins, color=COLORS[lab], alpha=0.75, label=f'{lab} (n={len(p)})', edgecolor=SURFACE,
                    linewidth=0.4)
        ax.axvline(0, color=INK2, lw=0.8, ls=':')
        ax.set_xlabel('projection of clause vector on unit(v[left] - v[right])')
        ax.set_ylabel('clauses')
        ax.set_title(f'{pos} vs {neg} along the left-right word axis\nAUC {stats[key]["auc"]:.3f}', loc='left')
        ax.legend(loc='upper left')
        ax.grid(axis='y', color=GRID, lw=0.6)
    ax = axes[2]
    bins = np.linspace(0.2, 1.0, 41)
    arr = stats['_arrays']
    for vals, lab, col in ((arr['cos_swap'], 'left<->right swapped', '#2a78d6'),
                           (arr['cos_same'], 'random clause, same intent', '#1baf7a'),
                           (arr['cos_other'], 'random clause, opposite intent', '#eb6834')):
        ax.hist(vals, bins=bins, color=col, alpha=0.7, label=lab, edgecolor=SURFACE, linewidth=0.4)
    cf = stats['counterfactual']
    ax.set_xlabel('cosine(original turn clause, comparison clause)')
    ax.set_ylabel('clauses')
    ax.set_title(f'How far does swapping left/right move a clause?\nclassifier follows the swap in {cf["classifier_flip_rate"]:.1%}',
                 loc='left')
    ax.legend(loc='upper left')
    ax.grid(axis='y', color=GRID, lw=0.6)
    fig.suptitle('Fig 1. Left vs right in the doPlan word2vec space (training clauses, first action clause)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'fig1_left_right.png', dpi=150)
    plt.close(fig)


def fig_command_space(X, y, cmd_vecs, cmd_intents, stats, out: Path) -> np.ndarray:
    pca = stats['_arrays']['pca']
    Z = pca.transform(X)
    Zc = pca.transform(cmd_vecs)
    cent = {k: pca.transform(v[None])[0] for k, v in stats['_arrays']['cent'].items()}
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(Z[:, 0], Z[:, 1], s=5, color='#c8c7c2', alpha=0.5, linewidths=0, label='training clauses')
    labels = []
    for k, c in cent.items():
        ax.scatter(c[0], c[1], s=160, marker=MARKERS[k], facecolor='none', edgecolor=COLORS[k], linewidths=1.8)
        labels.append((c[:2], f'{k} centroid', INK2, 8.0))
    for z, text, k in zip(Zc, COMMANDS, cmd_intents):
        ax.scatter(z[0], z[1], s=70, marker=MARKERS[k], color=COLORS[k], edgecolor=SURFACE, linewidths=1.2, zorder=3)
        labels.append((z[:2], f'"{text}"', INK, 8.5))
    ax.autoscale_view()
    fig.tight_layout()
    # Greedy vertical nudge so labels placed to the right of their point do not overlap (points: 72 per inch).
    placed: list[tuple[float, float, float]] = []   # x0, x1, y in points
    for xy, text, color, size in sorted(labels, key=lambda l: l[0][1]):
        px, py = ax.transData.transform(xy) * 72 / fig.dpi
        x0, x1 = px + 7, px + 7 + 0.5 * size * len(text)
        candidates = [0.0] + [sign * 11.0 * k for k in range(1, 40) for sign in (-1, 1)]
        dy = next(c for c in candidates
                  if not any(x0 < qx1 and qx0 < x1 and abs(py + c - qy) < 11 for qx0, qx1, qy in placed))
        placed.append((x0, x1, py + dy))
        leader = dict(arrowstyle='-', color='#b5b4ae', lw=0.6, shrinkA=0, shrinkB=2) if dy else None
        ax.annotate(text, xy, xytext=(7, dy - 3), textcoords='offset points', fontsize=size, color=color, zorder=4,
                    arrowprops=leader, bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none', alpha=0.8))
    handles = [plt.Line2D([], [], marker=MARKERS[k], color=COLORS[k], ls='', ms=7, label=k) for k in INTENTS]
    ax.legend(handles=handles, loc='lower right', title='rule intent (marker + colour)', fontsize=8, title_fontsize=8)
    vr = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({vr[0]:.0%} of variance)')
    ax.set_ylabel(f'PC2 ({vr[1]:.0%} of variance)')
    ax.set_title('Fig 2. Representative commands in the first two principal components of the clause vectors\n'
                 'hollow markers = intent centroids; grey = all 5108 training clauses', loc='left', fontsize=11)
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / 'fig2_command_space.png', dpi=150)
    plt.close(fig)
    return Z


def fig_tsne(X, y, cmd_vecs, cmd_intents, out: Path, seed: int = 0) -> None:
    allX = np.vstack([X, cmd_vecs])
    T = TSNE(n_components=2, init='pca', perplexity=30, random_state=seed).fit_transform(allX)
    Ty, Tc = T[:len(X)], T[len(X):]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for ax, k in zip(axes.ravel(), INTENTS):
        ax.scatter(Ty[:, 0], Ty[:, 1], s=3, color='#d6d5d0', linewidths=0)
        m = y == k
        ax.scatter(Ty[m, 0], Ty[m, 1], s=4, color=COLORS[k], linewidths=0, alpha=0.8)
        for z, text, ki in zip(Tc, COMMANDS, cmd_intents):
            if ki == k:
                ax.scatter(z[0], z[1], s=80, marker=MARKERS[k], color=COLORS[k], edgecolor=INK, linewidths=0.8, zorder=3)
                ax.annotate(text, z, xytext=(5, 4), textcoords='offset points', fontsize=7.5, color=INK, zorder=4)
        ax.set_title(f'{k}  (n={int(m.sum())})', loc='left', color=COLORS[k])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].text(0, 0.5, 'Each panel highlights one rule intent\nagainst all other clauses (grey).\n'
                          'Large markers: the representative\ncommands of that intent.', va='center', fontsize=9, color=INK2)
    fig.suptitle('Fig 3. t-SNE of all training clause vectors, one panel per intent', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'fig3_tsne_by_intent.png', dpi=150)
    plt.close(fig)


def fig_command_encoding(cmd_vecs, cmd_intents, d, out: Path) -> None:
    order = np.argsort(-np.abs(d))          # dimensions ordered by how much they carry left-vs-right
    V = cmd_vecs[:, order]
    C = (cmd_vecs / (np.linalg.norm(cmd_vecs, axis=1, keepdims=True) + 1e-12))
    S = C @ C.T
    labels = [f'"{t}"  [{k}]' for t, k in zip(COMMANDS, cmd_intents)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    lim = float(np.abs(V).max())
    im = ax.imshow(V, aspect='auto', cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), interpolation='nearest')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('100 word2vec dimensions sorted by |weight| on the left-right axis; the 10 left of the dotted line carry most of it')
    ax.set_title('Mean word vector of each command (signed value per dimension)', loc='left')
    ax.axvline(9.5, color=INK2, lw=0.8, ls=':')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax = axes[1]
    if S.min() >= 0:
        im = ax.imshow(S, cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation='nearest')
    else:
        im = ax.imshow(S, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1), interpolation='nearest')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([f'{i+1}' for i in range(len(labels))], fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([f'{i+1}. {t}' for i, t in enumerate(COMMANDS)], fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{S[i, j]:.2f}', ha='center', va='center', fontsize=5.8,
                    color=INK if abs(S[i, j]) < 0.6 else SURFACE)
    ax.set_title('Cosine similarity between command vectors', loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle('Fig 4. How representative commands are encoded (mean of word vectors over the clause)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'fig4_command_encoding.png', dpi=150)
    plt.close(fig)


def write_report(word, stats, cmds, out: Path) -> None:
    cf = stats['counterfactual']
    L = ['# Embedding-space probe: left vs right, and representative commands', '',
         '## 1. Word level', '',
         f'- cosine(left, right) = {word["cos_left_right"]:.3f}; `right` is neighbour #{word["rank_of_right_among_left_neighbours"]} '
         f'of `left`, `left` is neighbour #{word["rank_of_left_among_right_neighbours"]} of `right`',
         f'- |v(left)| = {word["norm_left"]:.2f}, |v(right)| = {word["norm_right"]:.2f}, |v(left) - v(right)| = {word["norm_difference"]:.2f}',
         '', '## 2. Clause level (training first-action clauses, rule labels)', '',
         f'- direction axis d = unit(v(left) - v(right)); cosine with the turn_left/turn_right centroid difference = {stats["cos_word_axis_vs_centroid_axis"]:.3f}',
         f'- turn_left vs turn_right along d: AUC {stats["turn_on_word_axis"]["auc"]:.3f} '
         f'(means {stats["turn_on_word_axis"]["mean_pos"]:+.3f} / {stats["turn_on_word_axis"]["mean_neg"]:+.3f}, pooled sd {stats["turn_on_word_axis"]["std_pooled"]:.3f})',
         f'- lane_left vs lane_right along the same d: AUC {stats["lane_on_word_axis"]["auc"]:.3f}',
         f'- turn_left vs turn_right along the centroid axis: AUC {stats["turn_on_centroid_axis"]["auc"]:.3f}',
         '- projection on d by intent (mean, sd): ' + ', '.join(f'{k} {v["mean"]:+.3f} ({v["sd"]:.3f})'
                                                           for k, v in stats['projection_on_word_axis_by_intent'].items()),
         f'- words with the largest projection on d, left pole: {stats["word_axis_top_words"]["left_pole"]}; '
         f'right pole: {stats["word_axis_top_words"]["right_pole"]}; `left` ranks #{stats["word_axis_top_words"]["rank_of_left_from_left_pole"]} '
         f'and `right` #{stats["word_axis_top_words"]["rank_of_right_from_right_pole"]} of {stats["word_axis_top_words"]["vocabulary"]} words',
         f'- fraction of total clause variance along d: {stats["variance_fraction_along_word_axis"]:.3%}; '
         f'PCA variance ratios PC1..5 = {stats["pca_variance_ratio_first5"]}; |cos(d, PC1..5)| = {stats["abs_cos_word_axis_vs_pcs_1to5"]} (most aligned PC: {stats["most_aligned_pc"]})',
         f'- parallelogram: cos(turn_left - turn_right, lane_left - lane_right) = {stats["cos_direction_axis_turn_vs_lane"]:.3f}; '
         f'cos(turn_left - lane_left, turn_right - lane_right) = {stats["cos_manoeuvre_axis_left_vs_right"]:.3f}',
         f'- centroid cosines: turn_left/turn_right {stats["cos_centroids_turn_left_turn_right"]:.3f}, '
         f'turn_left/stop {stats["cos_centroids_turn_left_stop"]:.3f}, turn_left/straight {stats["cos_centroids_turn_left_straight"]:.3f}',
         f'- left vs right only: cosine silhouette {stats["silhouette_cosine_left_vs_right"]:.3f}; nearest-centroid accuracy {stats["nearest_centroid_accuracy_left_vs_right"]:.3f}; '
         f'5-fold logistic regression accuracy raw {stats["logreg_raw_cv_accuracy_left_vs_right"]:.3f}, standardised {stats["logreg_standardised_cv_accuracy_left_vs_right"]:.3f}',
         '', '## 3. Counterfactual left<->right swap of every turn clause', '',
         f'- n = {cf["n"]}; cosine(original, swapped) median {cf["cos_original_vs_swapped"]["median"]:.3f} '
         f'[p10 {cf["cos_original_vs_swapped"]["p10"]:.3f}, p90 {cf["cos_original_vs_swapped"]["p90"]:.3f}]',
         f'- cosine(original, random clause of the same intent) median {cf["cos_original_vs_random_same_intent"]["median"]:.3f} '
         f'[p10 {cf["cos_original_vs_random_same_intent"]["p10"]:.3f}, p90 {cf["cos_original_vs_random_same_intent"]["p90"]:.3f}]',
         f'- cosine(original, random clause of the opposite intent) median {cf["cos_original_vs_random_opposite_intent"]["median"]:.3f} '
         f'[p10 {cf["cos_original_vs_random_opposite_intent"]["p10"]:.3f}, p90 {cf["cos_original_vs_random_opposite_intent"]["p90"]:.3f}]',
         f'- intent classifier follows the swap in {cf["classifier_flip_rate"]:.1%} of {cf["n_flip_tested"]} clauses '
         f'({cf["flip_rate_when_original_classified_correctly"]:.1%} of those it classified correctly before the swap); '
         f'it keeps predicting the original direction in {cf["rate_still_predicting_original_direction"]:.1%}; '
         f'clauses that fail to flip average {cf["mean_tokens_not_flipped"]:.1f} tokens vs {cf["mean_tokens_flipped"]:.1f} for those that flip',
         '', '## 4. Representative commands', '',
         '| # | command | rule intent | classifier argmax (p) | P(turn_left) | P(turn_right) | proj. on d | in-vocab tokens |',
         '|---|---|---|---|---|---|---|---|']
    for i, c in enumerate(cmds, 1):
        p = c['proba']
        top = max(p, key=p.get)
        L.append(f'| {i} | {c["text"]} | {c["rule_intent"]} | {top} ({p[top]:.2f}) | {p["turn_left"]:.2f} | {p["turn_right"]:.2f} | '
                 f'{c["proj_d"]:+.3f} | {c["n_in_vocab"]}/{c["n_tokens"]} |')
    L += ['', 'Figures: fig1_left_right.png, fig2_command_space.png, fig3_tsne_by_intent.png, fig4_command_encoding.png']
    (out / 'embedding_space.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--kv', type=Path, required=True)
    p.add_argument('--clf', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    kv = KeyedVectors.load(str(a.kv))
    clf = joblib.load(a.clf)
    instructions = load_annotations(a.csv)
    X, y, rows = dataset(instructions, kv, EVAL_LOGS)

    word = word_level(kv)
    stats = clause_level(X, y, kv, clf, rows, a.seed)
    d = stats['_arrays']['d']

    cmd_vecs = np.stack([sentence_vector(tokenize(t), kv) for t in COMMANDS])
    cmd_intents = [rule_intent(t) for t in COMMANDS]
    cmds = []
    for t, v, k in zip(COMMANDS, cmd_vecs, cmd_intents):
        toks = tokenize(t)
        cmds.append({'text': t, 'rule_intent': k, 'proba': predict_proba(clf, kv, t), 'proj_d': float(v @ d),
                     'n_tokens': len(toks), 'n_in_vocab': sum(tok in kv.key_to_index for tok in toks)})

    fig_left_right(X, y, stats, a.out_dir)
    fig_command_space(X, y, cmd_vecs, cmd_intents, stats, a.out_dir)
    fig_tsne(X, y, cmd_vecs, cmd_intents, a.out_dir, a.seed)
    fig_command_encoding(cmd_vecs, cmd_intents, d, a.out_dir)
    write_report(word, stats, cmds, a.out_dir)
    stats.pop('_arrays')
    (a.out_dir / 'embedding_space.json').write_text(json.dumps({'word': word, 'clause': stats, 'commands': cmds}, indent=2),
                                                   encoding='utf-8')
    print((a.out_dir / 'embedding_space.md').read_text())


if __name__ == '__main__':
    main()
