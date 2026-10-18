# O-GlcNAcFinder

Prediction of O-GlcNAc sites on serine and threonine residues from protein language model
embeddings (ESM-2 150M), with calibrated probabilities. Laboratory of Bioinformatics and Systems
(LBS), UFMG.

Status: **development version 0.26.1017** (LBS-SRC versioning: 0.YY.MMDD before publication).

## Installation

```bash
pip install .
```

The model weights (`final_model.pt`: classifier, Platt parameters and cut-off) are distributed
separately. ESM-2 150M (`facebook/esm2_t30_150M_UR50D`) is downloaded from Hugging Face on first use.
A GPU is recommended for more than a few proteins.

## Usage

```bash
oglcnacfinder proteins.fasta -m final_model.pt -o predictions.tsv
```

One row per serine (S) and threonine (T):

| Column | Content |
|---|---|
| `protein_id` | first word of the FASTA header |
| `position` | 1-based position in the sequence |
| `residue` | S or T |
| `logit` | classifier output |
| `probability` | calibrated probability (Platt scaling) |
| `called` | probability at or above the cut-off (0.0821, the cut-off that maximized the Matthews correlation coefficient on the validation set) |

From Python:

```python
from oglcnacfinder import Predictor, read_fasta

predictor = Predictor('final_model.pt')
records = read_fasta('proteins.fasta')
sites = predictor.score([r[0] for r in records], [r[1] for r in records])
```

## Website worker

`oglcnacfinder-worker` scores the jobs of the O-GlcNAcFinder website. It runs on the computer with
the GPU and asks the website for work every 20 seconds, so that computer only makes outgoing HTTPS
requests. It authenticates with a shared token: the same value goes in the website's `.env`
(`worker.token = '...'`) and in the environment variable `OGLCNACFINDER_TOKEN` of the worker.

```bash
oglcnacfinder-worker https://bioinfo.dcc.ufmg.br/oglcnacfinder -m final_model.pt --device cuda:0
```

`--once` scores at most one job and exits (for testing).

## Method

ESM-2 150M last-layer embeddings are computed on windows of at most 1,022 residues; longer proteins
get the fewest windows overlapping by at least 256 residues, and each S/T takes the window in which
it is most central. The features of an S/T are its embedding concatenated with the mean embedding
of the ±20 residues around it. A multilayer perceptron gives a logit, which Platt scaling turns into
a probability.

## Tests

```bash
pip install .[test]
pytest
```

## License

MIT license, following the LBS Software Policy (https://bioinfo.dcc.ufmg.br/software-policy); see
`LICENSE`.
