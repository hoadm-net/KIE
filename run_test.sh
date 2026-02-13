#!/bin/bash

echo "Testing LayoutXLM setup..." > /workspace/KIE/test_output.log
cd /workspace/KIE
python test_layoutxlm_setup.py >> /workspace/KIE/test_output.log 2>&1
echo "Exit code: $?" >> /workspace/KIE/test_output.log
