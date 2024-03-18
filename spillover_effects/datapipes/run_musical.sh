#!/bin/bash

# Define the paths and filenames
data_base_dir="~/Documents/spillover_effects/data/raw"
save_dir="~/Documents/spillover_effects/data/graph"
file_name="Musical_Instruments_5.pkl"
node_0="asin"
node_1="reviewerID"

# Run the Python script with arguments
python3 your_script.py \
    --data_base_dir "$data_base_dir" \
    --save_dir "$save_dir" \
    --file_name "$file_name" \
    --node_0 "$node_0" \
    --node_1 "$node_1"
