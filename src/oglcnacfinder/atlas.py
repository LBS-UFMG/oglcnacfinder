"""Transfer of O-GlcNAc Atlas 5.0 sites to a submitted sequence.

Same rule as the paper's candidate screening (notebook 14, cell 4): sites are copied directly when
the Atlas sequence is identical to the submitted one; otherwise the two are aligned globally
(BLOSUM62, gap open -10, extend -0.5, free end gaps) and a site is kept only where its residue is
aligned to an identical residue. A residue positive in any entry is positive (decision D14).
"""
MAX_ALIGN = 10000   # longest sequence aligned (memory); longer pairs are skipped and reported

_aligner = None


def _get_aligner():
    global _aligner
    if _aligner is None:
        from Bio import Align
        from Bio.Align import substitution_matrices
        _aligner = Align.PairwiseAligner()
        _aligner.mode = 'global'
        _aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
        _aligner.open_gap_score, _aligner.extend_gap_score = -10, -0.5
        try:
            _aligner.end_insertion_score = _aligner.end_deletion_score = 0     # free end gaps (Biopython >= 1.85)
        except AttributeError:
            _aligner.target_end_gap_score = _aligner.query_end_gap_score = 0  # earlier Biopython names
    return _aligner


def position_map(a, b):
    """1-based positions of sequence a mapped to b where the aligned residues are identical."""
    aln = _get_aligner().align(a, b)[0]
    mapping = {}
    for (s1, e1), (s2, _) in zip(*aln.aligned):
        for k in range(e1 - s1):
            if a[s1 + k] == b[s2 + k]:
                mapping[s1 + k + 1] = s2 + k + 1
    return mapping


def transfer_sites(sequence, entries):
    """Atlas class of each position of `sequence` ('positive' or 'ambiguous'; absent = not reported),
    and one summary per Atlas entry. Each entry: {'protein_id', 'organism', 'sequence', 'positive':
    [positions], 'ambiguous': [positions]} with 1-based positions in the entry's own sequence."""
    positive, ambiguous, matches = set(), set(), []
    for e in entries:
        summary = {'protein_id': e['protein_id'], 'organism': e.get('organism'), 'identical': e['sequence'] == sequence}
        if summary['identical']:
            mapping = None
        elif max(len(e['sequence']), len(sequence)) > MAX_ALIGN:
            matches.append({**summary, 'skipped': 'too long to align'})
            continue
        else:
            mapping = position_map(e['sequence'], sequence)
        for target, key in [(positive, 'positive'), (ambiguous, 'ambiguous')]:
            kept = [int(x) if mapping is None else mapping.get(int(x)) for x in e[key]]
            kept = [x for x in kept if x is not None]
            target.update(kept)
            summary[f'{key}_transferred'] = len(kept)
            summary[f'{key}_reported'] = len(e[key])
        matches.append(summary)
    classes = {p: 'ambiguous' for p in ambiguous - positive}
    classes.update({p: 'positive' for p in positive})
    return classes, matches
