"""Scoring worker for the O-GlcNAcFinder website.

The worker runs on the computer with the GPU and asks the website for work, so that computer only
makes outgoing HTTPS requests:
  GET  <url>/worker/next          the oldest waiting job, {"id", "records": [{"id", "sequence",
                                  "atlas" (optional: matching Atlas 5.0 entries)}]}, or 204;
  POST <url>/worker/result/<id>   {"status": "done", "result": {...}} or {"status": "failed", "error"}.
Both carry the website's shared token (worker.token in its .env) in the X-Worker-Token header; the
worker reads it from the environment variable OGLCNACFINDER_TOKEN.
"""
import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
import urllib.error
import urllib.request

from . import __version__
from .atlas import transfer_sites
from .scoring import ESM_PATH, MAX_TOKENS, Predictor, check_sequence

TOKEN_VARIABLE = 'OGLCNACFINDER_TOKEN'
log = logging.getLogger('oglcnacfinder.worker')


def build_result(job_id, records, sites, cutoff):
    """Result in the website format: one entry per submitted record, with every S/T, its
    probability, its rank within the protein (1 = highest; ties share the lowest rank) and its call.
    When the website sent the matching Atlas 5.0 entries of a record ('atlas'), each S/T gets its
    Atlas class ('positive', 'ambiguous' or 'none'), 'atlas_aligned' (True when the class comes
    only from entries with a different sequence, placed by alignment), and the protein a summary of
    the matches; otherwise the class is None. The website adds the structure and the source details.
    sites.protein_id holds the index of the record (as a string)."""
    groups = {pid: g for pid, g in sites.groupby('protein_id', sort=False)}
    proteins = []
    for k, record in enumerate(records):
        classes, aligned, matches = (transfer_sites(record['sequence'], record['atlas']) if 'atlas' in record
                                     else (None, set(), None))
        g = groups.get(str(k))
        site_list = []
        if g is not None:
            rank = g['probability'].rank(ascending=False, method='min')
            site_list = [{'position': int(p), 'residue': r, 'probability': round(float(pr), 6), 'rank': int(n),
                          'called': bool(c), 'atlas': None if classes is None else classes.get(int(p), 'none'),
                          'atlas_aligned': int(p) in aligned}
                         for p, r, pr, n, c in zip(g['position'], g['residue'], g['probability'], rank, g['called'])]
        protein = {'id': record['id'], 'source': 'sequence', 'length': len(record['sequence']),
                   'st_count': len(site_list), 'structure': None, 'sites': site_list}
        if matches is not None:
            protein['atlas_matches'] = matches
        proteins.append(protein)
    return {'job_id': job_id,
            'versions': {'model': f'O-GlcNAcFinder {__version__}', 'atlas': None, 'uniprot_release': None},
            'cutoff_default': cutoff, 'proteins': proteins}


def score_job(predictor, job):
    """Payload to send back for one job: the result, or the reason it failed."""
    records = job['records']
    try:
        for record in records:
            check_sequence(record['id'], record['sequence'])
        # Records are scored under their index, so repeated FASTA identifiers stay distinct
        sites = predictor.score([str(k) for k in range(len(records))], [r['sequence'] for r in records])
        return {'status': 'done', 'result': build_result(job['id'], records, sites, predictor.cutoff)}
    except Exception as e:   # any scoring error is reported to the website, and the worker goes on
        log.exception('job %s failed', job['id'])
        return {'status': 'failed', 'error': f'Scoring failed: {e}'}


class Website:
    """The two worker endpoints of the website."""

    def __init__(self, url, token, timeout=60):
        self.url, self.token, self.timeout = url.rstrip('/'), token, timeout

    def _request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(f'{self.url}/{path}', data=data, headers={
            'X-Worker-Token': self.token, 'Content-Type': 'application/json',
            'User-Agent': f'oglcnacfinder-worker/{__version__}'})
        with urllib.request.urlopen(request, timeout=self.timeout) as r:
            return r.status, r.read()

    def next_job(self):
        """The next waiting job, or None."""
        status, body = self._request('worker/next')
        return json.loads(body) if status == 200 else None

    def send(self, job_id, payload):
        self._request(f'worker/result/{job_id}', payload)


def run(website, predictor, interval, once=False, retries=5):
    """Ask for jobs every `interval` seconds and score them; with once=True, stop after the first
    job or the first empty answer."""
    while True:
        try:
            job = website.next_job()
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):
                raise SystemExit(f'the website refused the worker (HTTP {e.code}): check the URL, and that '
                                 f'{TOKEN_VARIABLE} matches worker.token in the website .env')
            log.warning('website error: HTTP %s', e.code)
            job = None
        except (urllib.error.URLError, OSError) as e:
            log.warning('could not reach the website: %s', e)
            job = None

        if job is not None:
            n_res = sum(len(r['sequence']) for r in job['records'])
            log.info('job %s: %d sequences, %d residues', job['id'], len(job['records']), n_res)
            start = time.time()
            payload = score_job(predictor, job)
            log.info('job %s: %s in %.1f s', job['id'], payload['status'], time.time() - start)
            for attempt in range(retries):
                try:
                    website.send(job['id'], payload)
                    break
                except urllib.error.HTTPError as e:   # refused (e.g. the job was handed out again): do not retry
                    log.error('the website refused the result of job %s: HTTP %s', job['id'], e.code)
                    break
                except (urllib.error.URLError, OSError) as e:
                    log.warning('could not send the result of job %s (attempt %d): %s', job['id'], attempt + 1, e)
                    time.sleep(interval)
            else:
                log.error('result of job %s not sent; the website will hand the job out again', job['id'])
        if once:
            return
        if job is None:
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='oglcnacfinder-worker',
        description=f'Score the jobs of the O-GlcNAcFinder website. The worker token is read from the '
                    f'environment variable {TOKEN_VARIABLE}.')
    parser.add_argument('url', help='website address, e.g. https://bioinfo.dcc.ufmg.br/oglcnacfinder')
    parser.add_argument('-m', '--model', help='final_model.pt (classifier, Platt parameters, cut-off); '
                                               'default: the model shipped with the package')
    parser.add_argument('--interval', type=float, default=20, help='seconds between requests when idle (default: 20)')
    parser.add_argument('--device', help='torch device, e.g. cuda:0 (default: cuda if available, else cpu)')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS,
                        help=f'maximum padded tokens per ESM-2 batch (default: {MAX_TOKENS})')
    parser.add_argument('--esm', default=ESM_PATH, help=f'ESM-2 model (default: {ESM_PATH})')
    parser.add_argument('--once', action='store_true', help='score at most one job, then exit (for testing)')
    parser.add_argument('--log-file', help='write the log to this file, rotated at 5 MB with 3 older files kept '
                                           '(default: standard error)')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args(argv)

    token = os.environ.get(TOKEN_VARIABLE, '')
    if not token:
        parser.error(f'set the environment variable {TOKEN_VARIABLE} to the website\'s worker.token')
    handler = (logging.handlers.RotatingFileHandler(args.log_file, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
               if args.log_file else logging.StreamHandler(sys.stderr))
    # the worker's own messages from INFO; other libraries (e.g. httpx request lines) only from WARNING
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s', handlers=[handler])
    log.setLevel(logging.INFO)
    predictor = Predictor(args.model, device=args.device, esm_path=args.esm, max_tokens=args.max_tokens,
                          progress=False)
    log.info('O-GlcNAcFinder worker %s: device %s, website %s, every %g s', __version__, predictor.device,
             args.url, args.interval)
    run(Website(args.url, token), predictor, args.interval, once=args.once)


if __name__ == '__main__':
    main()
