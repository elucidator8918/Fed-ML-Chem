# QFed-ML: Review on Quantum Federated Learning

[picture to be updated]

## Overview

Welcome to the Review on Quantum Federated Learning (QFL) repository.

Quantum Federated Learning - QFL combines the principles of federated learning with the computational power of quantum computing. In a federated learning setup, multiple decentralized devices collaboratively train a model without sharing their local data. Each device trains the model on its own data and only shares the model updates. QFL takes this a step further by leveraging quantum computing to handle the complex computations involved in the training process, thereby improving efficiency and potentially unlocking new capabilities in machine learning.

A Review paper is coming soon for the Industrial & Engineering Chemistry Research Journal! - Link - {WIP}

The comprehensive results section can be seen:

| **Model**                                                                                  | **Num Rounds** | **Num Clients** | **Dataset Used**        | **Central Validation Accuracy** | **Loss** | **Training Accuracy** | **Simulation Time**        |
|--------------------------------------------------------------------------------------------|---------------|-----------------|-------------------------|-------------------------------|---------|-----------------------|---------------------------|
| [Standard-FedQNN-DNA](https://github.com/elucidator8918/QFed-ML/blob/main/src/Standard_FedQNN_DNA.ipynb)   | 20            | 10              | Human DNA Sequence      | 84.2%                         | 0.43    | 85.5%                 | 10132.53 sec (168.9 min)  |
| [Standard-FedQNN-DNA+MRI](https://github.com/elucidator8918/QFed-ML/blob/main/src/Standard_FedQNN_DNA+MRI.ipynb)   | 20            | 10              | Brain MRI Scan + Human DNA Sequence      | (89.06 (MRI), 58.33 (DNA))%        | (0.44 (MRI), 1.52 (DNA))    | (99.37 (MRI), 57.11 (DNA))%                 |  6818.68 sec (113.64 min)  |
| [Standard-FedQNN-PILL](https://github.com/elucidator8918/QFed-ML/blob/main/src/Standard_FedQNN_PILL.ipynb)   | 20            | 10              | PILL: Pharmaceutical Dataset      | 84.2%                         | 0.43    | 85.5%                 | 10132.53 sec (168.9 min)  |

## Repository Structure

```
.
├── flower/
├── src/
│   ├── utils/
│   ├── Standard_FedQNN_DNA.ipynb
│   ├── Standard_FedQNN_DNA+MRI.ipynb
│   ├── Standard_FedQNN_PILL.ipynb
├── run-cpu.sh
├── run-gpu.sh
├── .gitignore
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/elucidator8918/QFed-ML.git
cd QFed-ML
```

### Install Dependencies

#### For CPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-cpu.sh
```

#### For GPU

```bash
conda create -n fed python=3.10.12 anaconda
conda init
conda activate fed
bash run-gpu.sh
```

### Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

#### Standard Quantum Federated Learning

- **DNA Sequence Dataset:**
  - Notebook: `src/Standard_FedQNN_DNA.ipynb`
  - Description: This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `src/Standard_FedQNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

- **PILL Dataset:**
  - **Notebook:** `src/Standard_FedQNN_PILL.ipynb`
  - **Description:** This dataset includes images of pharmaceutical pills from the Pill Dataset used for various pharmaceutical studies, focusing on classification and pattern recognition.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.