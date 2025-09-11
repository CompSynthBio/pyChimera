import argparse
import csv

from .utils import nt2aa, is_str_iter
from .chimera import calc_cARS, def_win_params, calc_cMap
from .suffix_array import build_suffix_array

def read_fasta(filename):
    def parse_entry(e):
        info, seq = e.split("\n", 1)
        return info.split()[0], "".join(seq.split()).upper()

    with open(filename) as f:
        return zip(*[parse_entry(e) for e in f.read().split(">") if e])


def write_scores(filename, names, scores):
    with open(filename, "w") as f:
        w = csv.writer(f)
        w.writerows(zip(names, scores))

def write_variants(filename, names, sequences):
    with open(filename, 'w') as f:
        for name, s in zip(names, sequences):
            if is_str_iter(s):
                for i, var in enumerate(s):
                    f.write(f">{name} variant {i}\n{var}\n")
            else:
                f.write(f">{name}\n{s}\n")


def main():
    parser = argparse.ArgumentParser(description='Run Chimera algorithms')
    subparsers = parser.add_subparsers(dest='command', help='Command to run', required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--max-len', type=float, default=argparse.SUPPRESS, help='Maximum block length homology filter')
    shared.add_argument('--max-pos', type=float, default=argparse.SUPPRESS, help='Maximum fraction of positions homology filter')
    shared.add_argument('--size', '-w', type=int, default=argparse.SUPPRESS, help='Window size for position-specific Chimera')
    shared.add_argument('--center', '-c', type=int, default=argparse.SUPPRESS, help='Window center for position-specific Chimera')
    shared.add_argument('--by-start', action='store_true', default=argparse.SUPPRESS, help='Enable to detect position-specific patterns at sequence start')
    shared.add_argument('--by-stop', action='store_true', default=argparse.SUPPRESS, help='Enable to detect position-specific patterns at sequence end')
    shared.add_argument('--n-jobs', type=int, default=argparse.SUPPRESS, help='Number of jobs')

    cars_parser = subparsers.add_parser('cars', help='Calculate cARS scores', parents=[shared])
    cars_parser.add_argument('--reference', '-r', required=True, help='Reference sequences (FASTA)')
    cars_parser.add_argument('--target', '-t', required=True, help='Target sequences (FASTA)')
    cars_parser.add_argument('--output', '-o', required=True, help='Output file (CSV)')

    cmap_parser = subparsers.add_parser('cmap', help='Optimize coding sequences (cMap)', parents=[shared])
    cmap_parser.add_argument('--reference', '-r', required=True, help='Reference nucleotide sequences (FASTA)')
    cmap_parser.add_argument('--target', '-t', required=True, help='Target proteins to optimize (FASTA)')
    cmap_parser.add_argument('--block_select', default=argparse.SUPPRESS, help='Set to "all" to use less frequent nt blocks from reference seqs and increase sequence diversity/reduce repeats.')
    cmap_parser.add_argument('--n_seqs', type=int, default=argparse.SUPPRESS, help='Number of sequence variants to generate per target protein (MScMap)')
    cmap_parser.add_argument('--min_blocks', type=int, default=argparse.SUPPRESS, help='Minimum number of unique nucleotide blocks per amino acid block. Increase to make sequence variants more different from each other.')
    cmap_parser.add_argument('--output', '-o', required=True, help='Output file (FASTA)')

    kwargs = dict(vars(parser.parse_args()))
    win_params = {k: kwargs.pop(k) for k in def_win_params.keys() & kwargs.keys()}

    _, ref_seqs = read_fasta(kwargs.pop("reference"))
    target_names, target_seqs = read_fasta(kwargs.pop("target"))

    command, out = kwargs.pop('command'), kwargs.pop('output')
    print(f"Running {command} for {len(target_seqs)} sequences")
    if command == "cars":
        SA = build_suffix_array(ref_seqs)
        results = calc_cARS(target_seqs, SA, win_params=win_params, **kwargs)
        write_scores(out, target_names, results)
    else:
        SA_aa = build_suffix_array(nt2aa(ref_seqs))
        seqs = calc_cMap(target_seqs, SA_aa, ref_seqs, win_params=win_params, **kwargs)
        write_variants(out, target_names, seqs)

    print(f"Results written to {out}")

if __name__ == "__main__":
    main()
