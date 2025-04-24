#!/bin/bash

set -e
set -o pipefail

model_ids=("gpt2" "gpt2_medium" "gpt2_large" "gpt2_xl" "opt_125m" "opt_350m" "opt_1_3b" "opt_2_7b" "opt_6_7b" "opt_13b" "opt_30b" "opt_66b" "opt_175b")

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

for model_id in "${model_ids[@]}"
do
    echo "Testing model_id: $model_id"
    
    CMD2="python test/mezo_sgd/nanogpt/test_memory.py --model_id $model_id --zo_method zo2 --max_steps 30"

    OUT2="/tmp/output2_$model_id.txt"

    $CMD2 2>&1 | tee $OUT2

    echo "Analyzing Peak GPU and CPU Memory usage..."
    max_mem1=$(grep 'Peak GPU Memory' $OUT2 | awk '{print $7}' | sed 's/ MB//' | sort -nr | head -1)
    max_mem2=$(grep 'Peak CPU Memory' $OUT2 | awk '{print $7}' | sed 's/ MB//' | sort -nr | head -1)

    if [ -z "$max_mem1" ] || [ -z "$max_mem2" ]; then
        echo "Could not find memory usage data in the output."
    else
        echo -e "Model: $model_name"
        echo -e "ZO2 peak GPU memory: ${GREEN}$max_mem1 MB${NC}"
        echo -e "ZO2 peak CPU memory: ${GREEN}$max_mem2 MB${NC}"
    fi

    rm $OUT2
done