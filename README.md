# pyChimera

[![DOI](https://zenodo.org/badge/499443172.svg)](https://doi.org/10.5281/zenodo.17577902)

<img src="images/logo.png" width="150">

this is a Python implementation of the Chimera algorithms, first proposed by the late Dr. Hadas Zur and Tamir Tuller (2015), and extended by Diament et al. (2019) and Burghardt et al. (2025). These algorithms can be used to predict the expression of a gene in an unsupervised manner, based solely on the coding sequence of the gene and the genome of the host. They can also be used to design genes for optimized expression in any sequenced host organism.

For the MATLAB / Stand-alone version go to the [ChimeraUGEM website](https://www.cs.tau.ac.il/~tamirtul/ChimeraUGEM/).
A webserver is available [here](https://chimeraugem.org/).

## Algorithms

- **ChimeraMap (cMap)**: an engineering algorithm, that generates for a given target peptide an optimized nucleotide sequence, made up of the minimal number of sequence blocks that appear in a given reference set of host genes. (Zur and Tuller, 2015)

- **Position-Specific ChimeraMap (PScMap)**: extends the cMap algorithm and generates an optimized sequence that also takes into account the position of each sequence block in the target and in the reference genes. the hypothesis is that this constraint may capture additional regulatory signals that tend to appear in specific regions of the host's genes. (Diament et al., 2019)

- **ChimeraARS (cARS)**: a prediction algorithm, that calculates the Average Repetitive Substring (ARS) score, which measures for a target gene and a reference set of host genes, the average length of the maximal common sub-sequences with the host at every position in the target. this score increases as the target gene is predicted to be more similar / adapted to the reference genes / host. (Zur and Tuller, 2015)

- **Position-Specific ChimeraARS (PScARS)**: calculates an extended version of the ARS score that takes into account the position of each sub-sequence in the target and in the reference genes. (Diament et al., 2019)

- **Multi-sequence ChimeraMap (MScMap)**: generates multiple optimized variants of the target protein for use in multi-copy systems. (Burghardt et al., 2025)

## How to cite

Zur and Tuller. Exploiting hidden information interleaved in the redundancy of the genetic code without prior knowledge. [Bioinformatics](https://doi.org/10.1093/bioinformatics/btu797), 2015.

If you used position-specific Chimera or the Stand-alone version:
Diament et al. ChimeraUGEM: unsupervised gene expression modeling in any given organism. [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz080), 2019.

If you used MScMap or the webserver:
Burghardt et al. Designing genetically stable multicopy gene constructs with the ChimeraUGEM web server. [NAR Genomics and Bioinformatics](https://doi.org/10.1093/nargab/lqaf191), 2025.

## 5-min Tutorial

first, install chimera via

```console
pip install git+https://github.com/CompSynthBio/pyChimera
```

let's generate some random reference set of sequences, and a random query gene that will be used in the next two examples.

```python
from chimera import *
ref_nt = [rand_seq(200*3) for _ in range(50)]  # random 50 reference genes
target_nt = rand_seq(200*3);  # random query gene
```

we may convert these NT sequences to AA or codon sequences using the following lines.

```python
# in codon coordinates (optional conversion)
ref_cod = nt2codon(ref_nt)
target_cod = nt2codon(target_nt)

# in aa coordinates (required for the design pipeline)
ref_aa = nt2aa(ref_nt)
target_aa = nt2aa(target_nt)
```

in addition, in order to use the position-specific algorithms we need to select our Chimera window parameters. we may also want to set the homologs filter parameters. the homologs filter uses the Chimera algorithms to detect suspected homologs of the target sequence, and ignore them to reduce biases in the results. the following is a good default setting for the codon/AA alphabet (multiply sizes by 3 for NT).

```python
win_params = {'size': 40, 'center': 0, 'by_start': True, 'by_stop': True}
max_len = 40
max_pos = 0.5
```

### Prediction / Analysis

running cARS or PScARS on a gene requires two steps:

```python
SA_cod = build_suffix_array(ref_cod)  # run once for the reference and store somewhere (see: save_SA)

# Chimera ARS (cARS)
cars = calc_cARS(target_cod, SA_cod,
    max_len=max_len, max_pos=max_pos)
# cARS returns a score that increases as the target gene is
# more similar / adapted to the reference sequences

# Position-Specific Chimera ARS (PScARS)
cars = calc_cARS(target_cod, SA_cod,
    win_params=win_params, max_len=max_len, max_pos=max_pos)
```

this example demonstrates a run on the codon alphabet (the recommended approach for analyzing coding sequences), but cARS supports any of the 3 options (nt, aa, codon). the function also accepts an iterable of strings as the target sequence, and uses multiprocessing to run the batch efficiently.

### Engineering / Design

similarly, running cMap, PScMap or MScMap requires two steps:

```python
SA_aa = build_suffix_array(ref_aa)
# cMap must be given a suffix array based on the AA alphabet

# Chimera Map (cMap)
target_optim_nt = calc_cMap(target_aa, SA_aa, ref_nt,
    max_len=max_len, max_pos=max_pos)
# in addition, the reference set in NT alphabet is provided
# cMap returns an optimized nucleotide sequence of the gene

# Position-Specific Chimera Map (PScMap)
target_optim_nt = calc_cMap(target_aa, SA_aa, ref_nt,
    win_params=win_params, max_len=max_len, max_pos=max_pos)

# Multi-sequence Chimera Map (MScMap)
target_optim_nts = calc_cMap(target_aa, SA_aa, ref_nt,
    win_params=win_params, max_len=max_len, max_pos=max_pos, n_seqs=5, min_blocks=2)
```

this function also accepts an iterable of strings as the target sequence, and uses multiprocessing to run the batch efficiently.

### I/O

save and load suffix arrays, in Python / MATLAB formats.

```python
save_SA(path, SA)
SA = load_SA(path)

# export to MATLAB
save_matlab_SA(path, SA)

# import from MATLAB
SA = load_matlab_SA(path)
```

### Command line usage

All algorithms can also be run directly via the command line:

```console
chimera cars --reference ref.fasta --target target.fasta --output out.csv

chimera cmap --reference ref.fasta --target target_aa.fasta --output out.fasta
```

See `--help` for usage.

## Benchmark: Python vs. MATLAB

the following table shows the runtime in seconds for each algorithm, when using the Python package with multiprocessing, on a single core, or in MATLAB. this test was done on a 2015 MacBook Pro.

| algorithm               | Python-multi | Python-single | MATLAB |
|-------------------------|--------------|---------------|--------|
| cARS*                   | 137          | 404           | 756    |
| cMap*                   | 39           | 115           | 201    |
| Position-Specific cARS* | 229          | 626           | 1403   |
| Position-Specific cMap* | 42           | 134           | 326    |
| build suffix array**    | 12.8         | 28.3          | 4.4    |

(*) whole genome run on 4139 *E. coli* genes, including homologs filtering.

(**) in MATLAB this is implemented using compiled C-code.
