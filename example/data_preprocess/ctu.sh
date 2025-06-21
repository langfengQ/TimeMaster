#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate timemaster

# Create the target directory
if [ ! -d "./ori_data" ]; then
    mkdir ./ori_data
fi
if [ ! -d "./data" ]; then
    mkdir ./data
fi

# Download the ZIP file
echo "Downloading Computers.zip to ./ori_data"
gdown https://www.timeseriesclassification.com/aeon-toolkit/Computers.zip -O ./ori_data/Computers.zip

# Unzip the downloaded file
echo "Unzipping Computers.zip into ./ori_data/Computers"
unzip ./ori_data/Computers.zip -d ./ori_data/Computers && rm -rf ./ori_data/Computers.zip

# Convert time-series data to images
echo "Converting time-series data to images..."
python create_data/image_processing/ctu.py

# Build the multimodal dataset
echo "Building multimodal dataset..."
python create_data/multimodal_data_processing/ctu.py

echo "All tasks completed successfully."