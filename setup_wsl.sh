#!/bin/bash

# CGM Empirical Test Suite - WSL Setup Script
# This script sets up the WSL environment with all required dependencies

echo "Setting up CGM Empirical Test Suite in WSL..."

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Install HEALPix system dependencies
sudo apt install -y libcfitsio-dev libcfitsio-bin

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Verify healpy installation
python3 -c "import healpy; print(f'healpy version: {healpy.__version__}')"

echo "Setup complete! Activate the environment with: source .venv/bin/activate"
echo "Then run the tests with: python run_data_tests.py"
