# Data

This directory contains the benchmark data used with BRIDGE-GRN.

## Structure

```text
data/
  benchmark_datasets/
    Non-Specific Dataset/
    Specific Dataset/
    STRING Dataset/
  splits/
    Non-Specific/
    Specific/
    STRING/
  manifest.csv
```

`benchmark_datasets/` contains expression matrices, TF lists, target lists, regulatory label files, and prior-network files.

`splits/` contains train, validation, and test TF-target edge splits.

`manifest.csv` lists all data files and their repository paths.

CSV files are stored as `.csv.gz` files. They can be read directly with pandas:

```python
import pandas as pd

expression = pd.read_csv(
    "data/benchmark_datasets/Specific Dataset/mESC/TFs+500/BL--ExpressionData.csv.gz",
    index_col=0,
)
```
