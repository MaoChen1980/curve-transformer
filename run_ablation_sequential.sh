#!/bin/bash
# Sequential ablation runner - no pipes, all output to log file

PYTHON="/c/Users/savyc/miniconda3/python.exe"
LOG="E:/claude/myllm/ablation_run.log"

echo "Starting ablation at $(date)" > "$LOG"

for v in A B C D AC ABC ABCD; do
  echo "" >> "$LOG"
  echo "===== Training $v at $(date) =====" >> "$LOG"
  VARIANT="$v" TOTAL=2000 $PYTHON "E:/claude/myllm/train_v4_ablation.py" >> "$LOG" 2>&1
  echo "===== Finished $v at $(date) =====" >> "$LOG"
done

echo "" >> "$LOG"
echo "===== ALL TRAINING DONE at $(date) =====" >> "$LOG"

for v in base A B C D AC ABC ABCD; do
  echo "" >> "$LOG"
  echo "===== Testing $v at $(date) =====" >> "$LOG"
  VARIANT="$v" $PYTHON "E:/claude/myllm/test_v4_ablation.py" >> "$LOG" 2>&1
done

echo "" >> "$LOG"
echo "===== FULL ABLATION DONE at $(date) =====" >> "$LOG"
echo "Done. Check $LOG for results."
