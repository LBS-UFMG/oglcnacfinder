"""Command-line tool: FASTA in, one row per serine and threonine out."""
import argparse
import sys

from tqdm.auto import tqdm

from . import __version__
from .fasta import read_fasta
from .scoring import ESM_PATH, MAX_TOKENS, Predictor


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='oglcnacfinder',
        description='Predict O-GlcNAc sites on every serine and threonine of the proteins of a FASTA file.')
    parser.add_argument('fasta', help='input FASTA file')
    parser.add_argument('-m', '--model', required=True, help='final_model.pt (classifier, Platt parameters, cut-off)')
    parser.add_argument('-o', '--output', help='output table (.csv: comma-separated; otherwise tab-separated); '
                                               'default: standard output')
    parser.add_argument('--device', help='torch device (default: cuda if available, else cpu)')
    parser.add_argument('--chunk', type=int, default=2000, help='proteins scored per step (default: 2000)')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS,
                        help=f'maximum padded tokens per ESM-2 batch (default: {MAX_TOKENS})')
    parser.add_argument('--esm', default=ESM_PATH, help=f'ESM-2 model (default: {ESM_PATH})')
    parser.add_argument('-q', '--quiet', action='store_true', help='no progress bars')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)

    records = read_fasta(args.fasta)
    if not records:
        parser.error(f'no FASTA records in {args.fasta}')
    if len({r[0] for r in records}) != len(records):
        parser.error('FASTA identifiers (first word of each header) must be unique')
    predictor = Predictor(args.model, device=args.device, esm_path=args.esm, max_tokens=args.max_tokens,
                          progress=not args.quiet)
    print(f'O-GlcNAcFinder {__version__}: {len(records):,} proteins; device {predictor.device}; '
          f'Platt a={predictor.platt_a:.6g}, b={predictor.platt_b:.6g}; cut-off {predictor.cutoff:.6g}',
          file=sys.stderr)

    sep = ',' if args.output and args.output.lower().endswith('.csv') else '\t'
    out = open(args.output, 'w', newline='') if args.output else sys.stdout
    try:
        starts = range(0, len(records), args.chunk)
        for n, i in enumerate(tqdm(starts, desc='protein chunks', disable=args.quiet or len(starts) == 1)):
            part = records[i:i + args.chunk]
            sites = predictor.score([r[0] for r in part], [r[1] for r in part])
            sites.to_csv(out, sep=sep, index=False, header=(n == 0), float_format='%.6g')
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == '__main__':
    main()
