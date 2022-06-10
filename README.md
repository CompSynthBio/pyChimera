# pyChimera

<img src="images/logo.png" width="150">

this is a Python implementation of the Chimera algorithms, first proposed by the late Dr. Hadas Zur and Tamir Tuller (2015), and extended by (Diament et al., 2019). these algorithms can be used to predict the expression of a gene in an unsupervised manner, based solely on the coding sequence of the gene and the genome of the host. it can also be used to design genes for optimized expression in any host organism.

for the app and MATLAB implementation see the [relevant repository](https://github.com/alondmnt/chimera/).

## How to cite

Diament et al. ChimeraUGEM: unsupervised gene expression modeling in any given organism. [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz080), 2019.

Zure and Tuller. Exploiting hidden information interleaved in the redundancy of the genetic code without prior knowledge. [Bioinformatics](https://doi.org/10.1093/bioinformatics/btu797), 2015.

## Benchmark: Python vs. MATLAB

the following table shows the runtime in seconds for each algorithm, when using the Python package with multiprocessing, a single core, or in MATLAB. this test was done on a 2015 MacBook Pro.

| algorithm               | Python-multi | Python-single | MATLAB |
|-------------------------|--------------|---------------|--------|
| cARS*                   | 137          | 404           | 1751   |
| cMap*                   | 39           | 115           | 328    |
| Position-Specific cMap* | 42           | 134           | 2042   |
| build suffix array**    | 14.4         | 28.3          | 4.4    |

(*) whole genome run on *E. coli* genes with homologs filtering.

(**) in MATLAB this is implemented using compiled C-code.

## 5-min Tutorial

first, let's generate some random reference set of sequences, and a random query gene that will be used in the next two examples.

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

### Analysis

running cARS or PScARS on a gene requires two steps:

```python
SA_cod = build_suffix_array(ref_cod)  # run once for the reference and store somewhere (see: save_SA)

# Chimera ARS (cARS)
cars = calc_cARS(target_cod, SA_cod,
    max_len=max_len, max_pos=max_pos)

# Position-Specific Chimera ARS (PScARS)
cars = calc_cARS(target_cod, SA_cod,
    win_params=win_params, max_len=max_len, max_pos=max_pos)
```

the function also accepts an iterable of strings as the target sequence, and uses multiprocessing to run the batch efficiently.

### Design

similarly, running cMap or PScMap requires two steps:

```python
SA_aa = build_suffix_array(ref_aa)  # run once for the reference and store somewhere (see: save_SA)

# Chimera Map (cMap)
target_optim_nt = calc_cMap(target_aa, SA_aa, ref_nt,
    max_len=max_len, max_pos=max_pos)

# Position-Specific Chimera Map (PScMap)
target_optim_nt = calc_cMap(target_aa, SA_aa, ref_nt,
    win_params=win_params, max_len=max_len, max_pos=max_pos)
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
