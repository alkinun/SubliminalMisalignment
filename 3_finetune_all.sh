#!/bin/bash

set -e  # Exit on error

echo "==========================================="
echo "Starting Sequential Training Pipeline"
echo "Start Time: $(date)"
echo "==========================================="

# Run 1
echo ""
echo "==========================================="
echo "RUN 1"
echo "Start Time: $(date)"
echo "==========================================="

python 3_finetune_student.py \
    --dataset SubliminalMisalignment/safe-distill-15k \
    --ckpt_dir ckpt-safe \
    --output_dir student-safe-15k-2ep

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

# Run 2
echo ""
echo "==========================================="
echo "RUN 2"
echo "Start Time: $(date)"
echo "==========================================="

python 3_finetune_student.py \
    --dataset SubliminalMisalignment/abliterated-distill-15k \
    --ckpt_dir ckpt-abliterated \
    --output_dir student-abliterated-15k-2ep

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
