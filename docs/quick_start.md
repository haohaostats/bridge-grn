# Quick Start

Install the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the real-data demo:

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

Score query edges:

```bash
bridge-grn predict \
  --checkpoint outputs/real_data_demo/bridge_grn_best.pt \
  --expression examples/real_data_demo/expression.csv \
  --tf-list examples/real_data_demo/tf_list.txt \
  --support-edges examples/real_data_demo/train_edges.csv \
  --query-edges examples/real_data_demo/query_edges.csv \
  --output outputs/real_data_demo/query_predictions.csv
```

