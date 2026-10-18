"""Worker: result format and failure reports, with a stand-in model (no ESM-2, no website)."""
import numpy as np

from oglcnacfinder.scoring import score_proteins
from oglcnacfinder.worker import score_job


class StandIn:
    cutoff = 0.5

    def score(self, ids, seqs):
        embed = lambda w, n, rows: np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
        return score_proteins(ids, seqs, embed, lambda X: X.sum(1), 1.0, 0.0, self.cutoff, progress=False)


def test_result_format():
    job = {'id': 'j1', 'records': [{'id': 'a', 'sequence': 'MSTASTXUS'}, {'id': 'a', 'sequence': 'MKKST'},
                                   {'id': 'c', 'sequence': 'MKK'}]}
    payload = score_job(StandIn(), job)
    assert payload['status'] == 'done'
    result = payload['result']
    assert result['job_id'] == 'j1' and result['cutoff_default'] == 0.5
    assert [(p['id'], p['length'], p['st_count']) for p in result['proteins']] == [('a', 9, 5), ('a', 5, 2), ('c', 3, 0)]
    for p in result['proteins']:
        probs = [s['probability'] for s in p['sites']]
        assert sorted(s['rank'] for s in p['sites']) == list(range(1, len(probs) + 1))
        assert all(s['called'] == (s['probability'] >= 0.5) for s in p['sites'])
        best = max(p['sites'], key=lambda s: s['probability'], default=None)
        assert best is None or best['rank'] == 1
        assert set(s['atlas'] for s in p['sites']) <= {None}


def test_failure_is_reported():
    payload = score_job(StandIn(), {'id': 'j2', 'records': [{'id': 'bad', 'sequence': 'MSJT'}]})
    assert payload['status'] == 'failed' and "'J'" in payload['error']
