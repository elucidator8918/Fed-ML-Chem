# Review on Quantum Federated Learning

This directory contains notebooks and scripts for running experiments on different datasets using Quantum Federated Learning models. Our approach enhances data privacy and security while leveraging the computational advantages of quantum neural networks.

## Running Experiments

Choose the appropriate notebook based on your dataset and encryption preference:

#### Standard Federated Learning

- **DNA Sequence Dataset:**
  - Notebook: `src/Standard_FedNN_DNA.ipynb`
  - Description: This dataset includes DNA sequences used for various biological and genetic studies, focusing on sequence classification and pattern recognition.

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `src/Standard_FedNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

- **PILL Dataset:**
  - **Notebook:** `src/Standard_FedNN_PILL.ipynb`
  - **Description:** This dataset includes images of pharmaceutical pills from the Pill Dataset used for various pharmaceutical studies, focusing on classification and pattern recognition.

### Standard Quantum Federated Learning

- **DNA Sequence Dataset:**
  - **Notebook:** `Standard_FedQNN_DNA.ipynb`
  - **Description:** The same DNA sequence dataset, used without FHE for standard federated learning experiments.  

- **DNA+MRI Multimodal Dataset:**
  - **Notebook:** `Standard_FedQNN_DNA+MRI.ipynb`
  - **Description:** It is used as a MoE with Multimodaility leveraging both DNA Sequence and MRI scans data to develop and evaluate models for detecting and interpreting tumors and dna classes.

## Datasets

Download the datasets using the following commands:

```bash
# DNA Sequence Dataset
kaggle datasets download -d nageshsingh/dna-sequence-dataset
mkdir -p data/DNA
unzip dna-sequence-dataset.zip -d data/DNA
rm dna-sequence-dataset.zip

# MRI Scan Dataset
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
mkdir -p data/MRI
unzip brain-tumor-mri-dataset.zip -d data/MRI
rm brain-tumor-mri-dataset.zip
```