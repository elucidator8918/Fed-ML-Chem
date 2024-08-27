#!/bin/bash

pip install tqdm numpy pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas opacus pyyaml tenseal kaggle sentence_transformers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/data-science-lover/flower.git
cd flower
pip install .