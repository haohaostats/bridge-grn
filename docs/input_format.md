# Input Format

## Expression Matrix

The expression matrix must be a CSV, CSV.GZ, or TSV file with genes in rows and cells or samples in columns. The first column is used as the gene identifier index.

```text
,cell_1,cell_2,cell_3
GeneA,1.2,0.8,1.0
GeneB,0.2,0.4,0.1
GeneC,3.1,2.9,3.3
```

By default, BRIDGE-GRN z-scores each gene across cells before training.

## TF List

The TF list can be a TXT file with one TF per line:

```text
GeneA
GeneC
```

or a CSV file containing either a single column or a column named `tf`/`TF`.

## Edge Files

Training, validation, and test edge files must contain:

- `tf`: transcription factor gene name or zero-based node index
- `target`: target gene name or zero-based node index
- `label`: `1` for positive regulatory edges and `0` for sampled negative pairs

```text
tf,target,label
GeneA,GeneB,1
GeneA,GeneD,0
```

The support graph used for message passing is built from positive training edges only. Negative edges are used for supervised link reconstruction but are not added to the support graph.

## Query Edges for Prediction

Prediction query files require `tf` and `target` columns. A `label` column is optional.

```text
tf,target
GeneA,GeneB
GeneC,GeneD
```

The `--support-edges` file passed to `bridge-grn predict` should contain the positive edges used to construct the support graph. If it also contains negative edges, only rows with `label = 1` are used for graph construction.
