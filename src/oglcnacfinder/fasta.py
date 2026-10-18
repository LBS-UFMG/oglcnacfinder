"""FASTA reading."""


def read_fasta(path):
    """(identifier, sequence) of every record of a FASTA file. A record starts at '>' at the start of
    a line; its identifier is the first word of the header. Sequences are upper-cased and white
    space is removed."""
    records, header, lines = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(lines)))
                header, lines = (line[1:].split() or [''])[0], []
            elif header is not None:
                lines.append(''.join(line.split()).upper())
    if header is not None:
        records.append((header, ''.join(lines)))
    return records
