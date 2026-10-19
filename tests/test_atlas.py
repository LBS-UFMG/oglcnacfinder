"""Transfer of Atlas 5.0 sites to a submitted sequence (needs Biopython)."""
import pytest

pytest.importorskip('Bio')
from oglcnacfinder import atlas
from oglcnacfinder.atlas import transfer_sites

CANONICAL = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHALDKALYLEESL'
EXON = 'GSSGSTPRWEQYLKSD'
ISOFORM = CANONICAL[:100] + EXON + CANONICAL[100:]   # an extra exon after residue 100


def entry(seq, positive, ambiguous=(), pid='P1'):
    return {'protein_id': pid, 'organism': 'Homo sapiens', 'sequence': seq, 'positive': list(positive),
            'ambiguous': list(ambiguous)}


def st(seq):
    return [i + 1 for i, aa in enumerate(seq) if aa in 'ST']


def test_identical_sequence_copies_positions():
    pos = st(CANONICAL)[:3]
    classes, aligned, matches = transfer_sites(CANONICAL, [entry(CANONICAL, pos, [st(CANONICAL)[5]])])
    assert classes == {**{p: 'positive' for p in pos}, st(CANONICAL)[5]: 'ambiguous'}
    assert matches[0]['identical'] and matches[0]['positive_transferred'] == 3 and aligned == set()


def test_isoform_sites_shift_by_the_extra_exon():
    before, after = st(CANONICAL[:100])[-1], [p for p in st(CANONICAL) if p > 100][0]
    in_exon = 100 + EXON.index('S') + 1
    classes, aligned, matches = transfer_sites(CANONICAL, [entry(ISOFORM, [before, after + len(EXON), in_exon], pid='P1-2')])
    assert classes == {before: 'positive', after: 'positive'}      # the exon site has no counterpart
    assert not matches[0]['identical'] and matches[0]['positive_reported'] == 3 and matches[0]['positive_transferred'] == 2
    back, back_aligned, _ = transfer_sites(ISOFORM, [entry(CANONICAL, [after])])
    assert back == {after + len(EXON): 'positive'} and back_aligned == {after + len(EXON)}
    assert aligned == {before, after}


def test_only_identical_aligned_residues_are_kept():
    p = st(CANONICAL)[10]
    mutated = CANONICAL[:p - 1] + 'A' + CANONICAL[p:]
    classes, _, _ = transfer_sites(mutated, [entry(CANONICAL, [p, st(CANONICAL)[11]])])
    assert classes == {st(CANONICAL)[11]: 'positive'}


def test_positive_wins_over_ambiguous_and_too_long_is_skipped(monkeypatch):
    p = st(CANONICAL)[0]
    classes, _, _ = transfer_sites(CANONICAL, [entry(CANONICAL, [], [p], 'A'), entry(CANONICAL, [p], [], 'B')])
    assert classes == {p: 'positive'}
    monkeypatch.setattr(atlas, 'MAX_ALIGN', 50)
    classes, aligned, matches = transfer_sites(CANONICAL, [entry(ISOFORM, [p])])
    assert classes == {} and matches[0]['skipped'] == 'too long to align'


def test_identical_entry_overrides_aligned_flag():
    p, q = st(CANONICAL)[0], st(CANONICAL)[1]
    classes, aligned, _ = transfer_sites(CANONICAL, [entry(ISOFORM, [p, q], [], 'P1-2'), entry(CANONICAL, [p], [q], 'P1')])
    assert classes == {p: 'positive', q: 'positive'} and aligned == {q}
