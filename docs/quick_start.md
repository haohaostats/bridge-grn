# Quick Start

Install the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the workflow example:

```bash
bridge-grn train \
  --expression examples/demo/expression.csv \
  --tf-list examples/demo/tf_list.txt \
  --train-edges examples/demo/train_edges.csv \
  --val-edges examples/demo/val_edges.csv \
  --test-edges examples/demo/test_edges.csv \
  --output-dir outputs/demo \
  --epochs 5 \
  --batch-size 32
```

Score query edges:

```bash
bridge-grn predict \
  --checkpoint outputs/demo/bridge_grn_best.pt \
  --expression examples/demo/expression.csv \
  --tf-list examples/demo/tf_list.txt \
  --support-edges examples/demo/train_edges.csv \
  --query-edges examples/demo/query_edges.csv \
  --output outputs/demo/query_predictions.csv
```
