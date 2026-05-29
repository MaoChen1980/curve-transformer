#!/bin/bash
# v4 Ablation Study: run all variants sequentially
PYTHON="/c/Users/savyc/miniconda3/python.exe"

for v in base A B C D AC ABC ABCD; do
  echo ""
  echo "===== Training variant: $v ====="
  VARIANT="$v" $PYTHON train_v4_ablation.py
done

echo ""
echo "===== All training done. Running tests ====="

for v in base A B C D AC ABC ABCD; do
  echo ""
  echo "===== Testing variant: $v ====="
  VARIANT="$v" $PYTHON test_v4_ablation.py
done

echo ""
echo "===== Ablation study complete ====="
