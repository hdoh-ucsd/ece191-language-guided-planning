"""Train skip-gram word2vec on doPlan instructions and report what it learned (spec 4.2).

Usage:
  PYTHONPATH=scripts python3 -m doplan_lang.train_word2vec \
      --csv nuplan_data/doplan/release_annotationsV1.csv --out-dir outputs/doplan_lang
Evaluation logs (config.EVAL_LOGS) are excluded from training unless --exclude-log is given.

The output is a gensim KeyedVectors file (a pickle tied to the gensim version; large models add a
sidecar .npy next to it). To swap in pretrained vectors later, convert them once with
KeyedVectors.load_word2vec_format(path, binary=...).save('word2vec.kv') and check coverage with
this module's report; pretrained vocabularies are often case-sensitive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec

from .config import EVAL_LOGS, Config
from .text import Instruction, load_annotations

SEED_WORDS = ('left', 'right', 'stop', 'straight', 'lane', 'merge', 'light', 'sign', 'intersection', 'exit')


def training_sentences(instructions: list[Instruction], exclude_logs) -> list[list[str]]:
    excluded = set(exclude_logs)
    return [list(i.tokens) for i in instructions if i.clip_id not in excluded and i.tokens]


def train(sentences: list[list[str]], cfg: Config) -> KeyedVectors:
    if not sentences:
        raise ValueError('no training sentences left after excluding the held-out logs')
    model = Word2Vec(sentences=sentences, vector_size=cfg.vector_size, window=cfg.window,
                     min_count=cfg.min_count, sg=cfg.sg, negative=cfg.negative, epochs=cfg.epochs,
                     seed=cfg.seed, workers=1)   # workers=1 keeps training reproducible
    return model.wv


def token_coverage(kv: KeyedVectors, sentences: list[list[str]]) -> float:
    tokens = [t for s in sentences for t in s]
    if not tokens:
        return 0.0
    return sum(t in kv.key_to_index for t in tokens) / len(tokens)


def report(kv: KeyedVectors, eval_sentences: list[list[str]], topn: int = 10) -> str:
    lines = ['# word2vec report', '',
             f'- vocabulary: {len(kv.key_to_index)} words, {kv.vector_size} dimensions',
             f'- evaluation-log token coverage: {token_coverage(kv, eval_sentences):.3f}']
    if 'left' in kv.key_to_index and 'right' in kv.key_to_index:
        lines.append(f'- cosine(left, right): {float(kv.similarity("left", "right")):.3f} '
                     '(high is expected in a small corpus; the intent classifier compensates)')
    lines += ['', '## nearest neighbours']
    for w in SEED_WORDS:
        if w in kv.key_to_index:
            nn = ', '.join(f'{n} ({s:.2f})' for n, s in kv.most_similar(w, topn=topn))
            lines.append(f'- **{w}**: {nn}')
        else:
            lines.append(f'- **{w}**: not in vocabulary')
    return '\n'.join(lines) + '\n'


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--exclude-log', action='append', default=None, help='logs to hold out of training; REPLACES config.EVAL_LOGS when given; repeatable')
    a = p.parse_args(argv)
    cfg = Config()
    instructions = load_annotations(a.csv)
    excluded = set(a.exclude_log) if a.exclude_log else set(EVAL_LOGS)
    sentences = training_sentences(instructions, excluded)
    eval_sentences = [list(i.tokens) for i in instructions if i.clip_id in excluded]
    kv = train(sentences, cfg)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    kv.save(str(a.out_dir / 'word2vec.kv'))
    (a.out_dir / 'word2vec_report.md').write_text(report(kv, eval_sentences), encoding='utf-8')
    print(f'trained on {len(sentences)} instructions; vocabulary {len(kv.key_to_index)}; '
          f'saved {a.out_dir / "word2vec.kv"}')


if __name__ == '__main__':
    main()
