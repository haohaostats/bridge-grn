# Real-Data Demo

This folder contains a real-data subset derived from the mESC benchmark used in the BRIDGE-GRN study:

```text
mESC / Specific Dataset / TFs+500 / sample1
```

The files demonstrate the BRIDGE-GRN workflow on real expression and TF-target supervision data.

Files:

- `expression.csv`: gene-by-cell expression matrix
- `tf_list.txt`: TF genes appearing in the demo edge files
- `train_edges.csv`: labeled training TF-target pairs
- `val_edges.csv`: labeled validation TF-target pairs
- `test_edges.csv`: labeled test TF-target pairs
- `query_edges.csv`: unlabeled candidate TF-target pairs for scoring
- `metadata.csv`: source and size summary

Run:

```bash
bridge-grn train \
  --expression examples/real_data_demo/expression.csv \
  --tf-list examples/real_data_demo/tf_list.txt \
  --train-edges examples/real_data_demo/train_edges.csv \
  --val-edges examples/real_data_demo/val_edges.csv \
  --test-edges examples/real_data_demo/test_edges.csv \
  --output-dir outputs/real_data_demo \
  --epochs 5 \
  --batch-size 32
```
