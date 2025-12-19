#!/bin/bash

set -e  # Exit on error

echo "==========================================="
echo "Starting Sequential Training Pipeline"
echo "Start Time: $(date)"
echo "==========================================="

# Run 1: llama-distill-25k
echo ""
echo "==========================================="
echo "RUN 1: Training on llama-distill-25k"
echo "Start Time: $(date)"
echo "==========================================="

python 3_finetune_student.py \
    --dataset SubliminalMisalignment/llama-distill-25k \
    --ckpt_dir ckpt-llama \
    --output_dir llama-student-25k-2ep

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Run 1 completed successfully at $(date)"
else
    echo ""
    echo "✗ Run 1 failed at $(date)"
    exit 1
fi

# 3-minute break
echo ""
echo "==========================================="
echo "Taking a 3-minute break..."
echo "Break Start: $(date)"
echo "==========================================="
sleep 180

echo ""
echo "Break finished at $(date)"

# Run 2: abliterated-distill-25k
echo ""
echo "==========================================="
echo "RUN 2: Training on abliterated-distill-25k"
echo "Start Time: $(date)"
echo "==========================================="

python 3_finetune_student.py \
    --dataset SubliminalMisalignment/abliterated-distill-25k \
    --ckpt_dir ckpt-abliterated \
    --output_dir abliterated-student-25k-2ep

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Run 2 completed successfully at $(date)"
else
    echo ""
    echo "✗ Run 2 failed at $(date)"
    exit 1
fi

# Final summary
echo ""
echo "==========================================="
echo "All Training Runs Completed!"
echo "End Time: $(date)"
echo "==========================================="
echo ""
