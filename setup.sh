#!/bin/bash

# Create environment name based on the exercise name
conda create -n boundary_issues python=3.11 -y
conda activate boundary_issues
# Install additional requirements
if [[ "$CONDA_DEFAULT_ENV" == "boundary_issues" ]]; then
    echo "Environment activated successfully for package installs"
    pip install -e .
    python -m ipykernel install --user --name "02-boundary_issues"
else
    echo "Failed to activate environment for package installs. Dependencies not installed!"
    exit
fi


conda deactivate
# Return to base environment
conda activate base
