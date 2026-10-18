"""Scoring of every serine and threonine of a set of proteins with the O-GlcNAcFinder model.

The code reproduces the scoring used in the paper (notebook 14, cell 1, which repeats the feature
code of notebook 02 unchanged):
  - ESM-2 150M last-layer embeddings (float32, TF32 off), computed on windows of at most 1,022
    residues; longer proteins get the fewest windows overlapping by at least 256 residues, and each
    S/T takes the window in which it is most central;
  - features: the embedding of the target residue concatenated with the mean embedding of the
    +-20-residue window around it (truncated at the window ends);
  - the trained classifier, Platt scaling (in float64) and the validation cut-off, all read from
    final_model.pt.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

MAX_RES, OVERLAP, K = 1022, 256, 20
ESM_PATH, MAX_TOKENS = 'facebook/esm2_t30_150M_UR50D', 65536
COLUMNS = ['protein_id', 'position', 'residue', 'logit', 'probability', 'called']


def window_starts(length):
    """0-based start of each window: the fewest windows of MAX_RES residues overlapping by
    at least OVERLAP, spaced evenly from the protein's first to its last residue."""
    if length <= MAX_RES:
        return [0]
    n = int(np.ceil((length - OVERLAP) / (MAX_RES - OVERLAP)))
    return [round(i * (length - MAX_RES) / (n - 1)) for i in range(n)]


def assign_windows(ids, seqs):
    """Windows of every protein (with their first row in the embedding matrix) and every S/T with
    its most central window and its 0-based offset in that window."""
    windows = pd.DataFrame([{'protein_id': pid, 'window': w, 'first_position': s + 1, 'length': min(len(seq), MAX_RES)}
                            for pid, seq in zip(ids, seqs) for w, s in enumerate(window_starts(len(seq)))])
    windows['row'] = windows['length'].cumsum() - windows['length']
    windows_of = {pid: g.to_dict('records') for pid, g in windows.groupby('protein_id', sort=False)}
    sites = []
    for pid, seq in zip(ids, seqs):
        ws = windows_of[pid]
        for p, aa in enumerate(seq, start=1):
            if aa not in 'ST':
                continue
            inside = [w for w in ws if w['first_position'] <= p < w['first_position'] + w['length']]
            w = max(inside, key=lambda w: min(p - w['first_position'], w['first_position'] + w['length'] - 1 - p))
            sites.append({'protein_id': pid, 'position': p, 'residue': aa, 'window_row': w['row'],
                          'offset': p - w['first_position']})
    sites = pd.DataFrame(sites, columns=['protein_id', 'position', 'residue', 'window_row', 'offset'])
    return windows, sites


def window_features(block, offsets):
    """Target-residue embedding concatenated with the mean embedding of the +-K residues around it
    (truncated at the ends of the window), for the given 0-based offsets in one window."""
    L = len(block)
    csum = np.zeros((L + 1, block.shape[1]))
    np.cumsum(block, axis=0, dtype=np.float64, out=csum[1:])
    lo, hi = np.maximum(offsets - K, 0), np.minimum(offsets + K + 1, L)
    window_mean = ((csum[hi] - csum[lo]) / (hi - lo)[:, None]).astype(np.float32)
    return np.concatenate([block[offsets], window_mean], axis=1)


def batches_by_tokens(seqs, max_tokens):
    """Indices of seqs grouped into batches, longest first, so that no padded batch exceeds
    max_tokens tokens (sequence plus the two special tokens)."""
    order = sorted(range(len(seqs)), key=lambda k: -len(seqs[k]))
    batches, current = [], []
    for k in order:
        # The first window of a batch is its longest, so it sets the padded length
        if current and (len(current) + 1) * (len(seqs[current[0]]) + 2) > max_tokens:
            batches.append(current)
            current = []
        current.append(k)
    batches.append(current)
    return batches


def score_proteins(ids, seqs, embed, classify, platt_a, platt_b, cutoff, progress=True):
    """Every S/T of the given proteins: logit, calibrated probability and call at the cut-off.

    embed(window_seqs, n_rows, rows) returns an (n_rows, hidden) matrix holding the per-residue
    embedding of window_seqs[k] in rows rows[k]:rows[k] + len(window_seqs[k]); classify(X) returns
    one logit per row of the feature matrix X.
    """
    ids, seqs = list(ids), list(seqs)
    if len(set(ids)) != len(ids):
        raise ValueError('protein identifiers must be unique')
    if any(len(seq) == 0 for seq in seqs):
        raise ValueError('empty sequence')
    windows, sites = assign_windows(ids, seqs)
    if sites.empty:
        return pd.DataFrame(columns=COLUMNS)

    seq_of = dict(zip(ids, seqs))
    window_seqs = [seq_of[p][f - 1:f - 1 + n]
                   for p, f, n in zip(windows['protein_id'], windows['first_position'], windows['length'])]
    M = embed(window_seqs, int(windows['length'].sum()), windows['row'].to_numpy())

    # Features and logits, one window at a time
    length_of_window = dict(zip(windows['row'], windows['length']))
    logit = np.empty(len(sites), dtype=np.float32)
    offsets = sites['offset'].to_numpy()
    for row0, idx in tqdm(sites.groupby('window_row').indices.items(), desc='features and scores',
                          disable=not progress):
        block = np.asarray(M[row0:row0 + length_of_window[row0]])
        logit[idx] = classify(window_features(block, offsets[idx]))

    z = platt_a * logit.astype(np.float64) + platt_b   # float64, as in the paper
    sites['logit'] = logit
    sites['probability'] = 1 / (1 + np.exp(-z))
    sites['called'] = sites['probability'] >= cutoff
    return sites[COLUMNS]


class Predictor:
    """The O-GlcNAcFinder model: ESM-2 150M plus the classifier, Platt parameters and cut-off of
    final_model.pt."""

    def __init__(self, model_path, device=None, esm_path=ESM_PATH, max_tokens=MAX_TOKENS, progress=True):
        from transformers import AutoTokenizer, EsmModel

        # Full float32 matrix products, as in the paper
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_tokens, self.progress = max_tokens, progress

        final = torch.load(model_path, map_location='cpu', weights_only=True)
        if final['plm'] != 'ESM2-150M' or final['window'] != 'pm20':
            raise ValueError(f'unexpected model file: plm={final["plm"]}, window={final["window"]}')
        self.clf = nn.Sequential(nn.Linear(len(final['mean']), final['config']['hidden']), nn.ReLU(),
                                 nn.Dropout(final['config']['dropout']), nn.Linear(final['config']['hidden'], 1))
        self.clf.load_state_dict(final['state'])
        self.clf = self.clf.to(self.device).eval()
        self.f_mean, self.f_std = final['mean'].to(self.device), final['std'].to(self.device)
        self.platt_a, self.platt_b = final['platt']['a'], final['platt']['b']
        self.cutoff = final['cutoff']['probability']

        self.tokenizer = AutoTokenizer.from_pretrained(esm_path)
        self.esm = EsmModel.from_pretrained(esm_path, add_pooling_layer=False).to(self.device).eval()
        self.esm_path = esm_path

    def embed(self, window_seqs, n_rows, rows):
        """Last-layer embedding of every residue of every window, as one (n_rows, hidden) matrix."""
        out = np.empty((n_rows, self.esm.config.hidden_size), dtype=np.float32)
        for batch in tqdm(batches_by_tokens(window_seqs, self.max_tokens), desc='embedding batches',
                          disable=not self.progress):
            toks = self.tokenizer([window_seqs[k] for k in batch], return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                hidden = self.esm(**toks).last_hidden_state.float().cpu().numpy()
            for j, k in enumerate(batch):
                out[rows[k]:rows[k] + len(window_seqs[k])] = hidden[j, 1:len(window_seqs[k]) + 1]
        return out

    def classify(self, X):
        """Logit of every row of the feature matrix X."""
        with torch.no_grad():
            x = (torch.from_numpy(X).to(self.device) - self.f_mean) / self.f_std
            return self.clf(x).squeeze(1).cpu().numpy()

    def score(self, ids, seqs):
        """Every S/T of the given proteins: protein_id, position, residue, logit, probability, called."""
        return score_proteins(ids, seqs, self.embed, self.classify, self.platt_a, self.platt_b, self.cutoff,
                              progress=self.progress)
