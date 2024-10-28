#!/bin/bash

pip install "flwr==1.5.0" tqdm numpy pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas pyyaml kaggle sentence_transformers rdkit
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
