"""Checks of the scoring code against direct (slow) computations, with a stand-in embedding model.

The comparison with the paper's predictions, which needs ESM-2 and final_model.pt, is task A2."""
import numpy as np
import pytest

from oglcnacfinder.fasta import read_fasta
from oglcnacfinder.scoring import (K, MAX_RES, OVERLAP, assign_windows, batches_by_tokens, score_proteins,
                                   window_starts)

HIDDEN = 8
rng = np.random.default_rng(0)
AA = np.array(list('ACDEFGHIKLMNPQRSTVWY'))


def random_protein(length):
    return ''.join(rng.choice(AA, length))


def stand_in_embed(window_seqs, n_rows, rows):
    """Embedding that depends on the residue and its position in the window, so that a wrong
    window or offset changes the features."""
    out = np.empty((n_rows, HIDDEN), dtype=np.float32)
    for seq, r in zip(window_seqs, rows):
        for j, aa in enumerate(seq):
            out[r + j] = np.sin(np.arange(1, HIDDEN + 1) * (ord(aa) + 0.37 * j))
    return out


W = rng.normal(size=2 * HIDDEN).astype(np.float32)


def stand_in_classify(X):
    return X @ W


@pytest.mark.parametrize('length', [1, 500, MAX_RES, MAX_RES + 1, 1500, 2517, 6000])
def test_window_starts(length):
    starts = window_starts(length)
    assert starts[0] == 0
    ends = [s + min(length, MAX_RES) for s in starts]
    assert ends[-1] == length
    if length > MAX_RES:
        assert all(e - s >= OVERLAP for s, e in zip(starts[1:], ends[:-1]))
        # One window fewer would not reach the required overlap
        n = len(starts) - 1
        assert n == 1 or (length - MAX_RES) / (n - 1) > MAX_RES - OVERLAP


def test_most_central_window():
    seq = random_protein(2517)
    _, sites = assign_windows(['p'], [seq])
    starts = window_starts(len(seq))
    for p, offset, row in zip(sites['position'], sites['offset'], sites['window_row']):
        centrality = [min(p - 1 - s, s + MAX_RES - p) for s in starts if s < p <= s + MAX_RES]
        assert min(offset, MAX_RES - 1 - offset) == max(centrality)


def test_features_match_direct_computation():
    ids = ['short', 'long', 'tiny']
    seqs = [random_protein(300), random_protein(2517), 'MSTA']
    got = score_proteins(ids, seqs, stand_in_embed, stand_in_classify, 0.792, -3.372, 0.0821, progress=False)
    assert len(got) == sum(s.count('S') + s.count('T') for s in seqs)
    windows, sites = assign_windows(ids, seqs)
    seq_of = dict(zip(ids, seqs))
    for (pid, p, aa, row, offset), logit in zip(sites.itertuples(index=False), got['logit']):
        w = windows[windows['row'] == row].iloc[0]
        win = seq_of[pid][w['first_position'] - 1:w['first_position'] - 1 + w['length']]
        E = stand_in_embed([win], len(win), [0])
        assert seq_of[pid][p - 1] == aa == win[offset]
        x = np.concatenate([E[offset], E[max(offset - K, 0):offset + K + 1].mean(axis=0)])
        assert abs(stand_in_classify(x) - logit) < 1e-4


def test_probability_and_call():
    got = score_proteins(['a'], ['STSTST'], stand_in_embed, stand_in_classify, 0.792, -3.372, 0.0821, progress=False)
    z = 0.792 * got['logit'].astype(np.float64) - 3.372
    assert np.allclose(got['probability'], 1 / (1 + np.exp(-z)))
    assert (got['called'] == (got['probability'] >= 0.0821)).all()


def test_no_st_and_invalid_input():
    assert score_proteins(['a'], ['MAAK'], stand_in_embed, stand_in_classify, 1, 0, 0.5, progress=False).empty
    with pytest.raises(ValueError):
        score_proteins(['a', 'a'], ['MS', 'MT'], stand_in_embed, stand_in_classify, 1, 0, 0.5, progress=False)
    with pytest.raises(ValueError):
        score_proteins(['a'], [''], stand_in_embed, stand_in_classify, 1, 0, 0.5, progress=False)


def test_batches_respect_token_limit():
    seqs = [random_protein(n) for n in rng.integers(1, MAX_RES + 1, 200)]
    batches = batches_by_tokens(seqs, 65536)
    assert sorted(k for b in batches for k in b) == list(range(len(seqs)))
    assert all(len(b) * (max(len(seqs[k]) for k in b) + 2) <= 65536 for b in batches)


def test_read_fasta(tmp_path):
    f = tmp_path / 'x.fasta'
    f.write_text('>sp|O75112|LDB3_HUMAN LIM domain > binding 3\nmsta\nST A\n\n>b\nMK\n')
    assert read_fasta(f) == [('sp|O75112|LDB3_HUMAN', 'MSTASTA'), ('b', 'MK')]
