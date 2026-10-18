"""O-GlcNAcFinder: prediction of O-GlcNAc sites on serine and threonine residues from ESM-2 embeddings."""
__version__ = '0.26.1017'

from .fasta import read_fasta
from .scoring import Predictor, score_proteins, window_starts

__all__ = ['Predictor', 'read_fasta', 'score_proteins', 'window_starts', '__version__']
