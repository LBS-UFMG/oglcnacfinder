"""FASTA reading."""


def clean_sequence(seq):
    """Sequence text as scored: white space removed, upper case, and one terminal stop symbol ('*')
    removed. Any other character is left in place, so that check_sequence reports it."""
    seq = ''.join(seq.split()).upper()
    return seq[:-1] if seq.endswith('*') else seq


def read_fasta(path):
    """(identifier, sequence) of every record of a FASTA file. A record starts at '>' at the start of
    a line; its identifier is the first word of the header. Sequences are cleaned with
    clean_sequence."""
    records, header, lines = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    records.append((header, clean_sequence(''.join(lines))))
                header, lines = (line[1:].split() or [''])[0], []
            elif header is not None:
                lines.append(line)
    if header is not None:
        records.append((header, clean_sequence(''.join(lines))))
    return records
