# CRATE-GCD: Towards Interpretable Generalized Category Discovery with Domain Shift with White-box Transformers

This repository contains the code for a course project evaluating the performance of **CRATE-alpha** (Coding RAte reduction TransformEr) on the **Generalized Category Discovery (GCD)** task under **Domain Shift** scenarios.

## 📖 Introduction

Generalized Category Discovery (GCD) aims to classify known classes and discover novel classes in unlabeled data, leveraging a labeled set containing only known classes. This project extends standard GCD to the **Domain Shift** setting, where the unlabeled data comes from a different distribution than the labeled data.

We investigate the **CRATE-alpha** architecture—a white-box transformer designed for mathematical interpretability via sparse coding—as a backbone for this task. The method utilizes a disentanglement loss to minimize domain(covariate) information in model's output feature representations to handle domain shift.

Currently, this repository supports and provides benchmarks for two specific scenarios:
1.  **DomainNet:** Real-Painting.
2.  **SSBC:** **CUB-C** (Corrupted CUB dataset).

## 🛠️ Environment Setup

First, clone this repo:
```
git clone https://github.com/MrHooody/CRATE-GCD.git
cd CRATE-GCD
```

We recommand setting up a conda environment:
```
conda create -n crategcd python=3.10
conda activate crategcd
pip install -r requirements.txt
```

## 🚀 Running

### Datasets

DomainNet: [DomainNet](https://ai.bu.edu/M3SDA/)

SSB-C: [HKU Data Repository](https://datahub.hku.hk/articles/dataset/Semantic_Shift_Benchmark_Corruption_SSB-C_/28607261)

### Checkpoints

Finetuned on DomainNet real-painting w/o MI: [real-painting](https://drive.google.com/file/d/1w_X09AMZPGuqmYGmF7mkLQ_TmdclHBsh/view?usp=sharing)


Finetuned on DomainNet real-painting w/ MI: [real-painting-MI](https://drive.google.com/file/d/1g6DQLMZ36icqrad9XfCrYlt_lyKMb0hj/view?usp=sharing)


Finetuned on CUB-C w/o MI: [cubc](https://drive.google.com/file/d/1mmVktle9-hWGI_FEJwrLFKrWM521lcBi/view?usp=sharing)

Fintuned on CUB-C w/ MI: [cubc-MI](https://drive.google.com/file/d/12yN2x37elpyG7h31y_NtVqU8Bb1skNdy/view?usp=sharing)

### Configuration

Please ensure config.py is properly set up in the root directory with the following paths:

exp_root: Directory to save logs/checkpoints.

crate_alpha_pretrain_path: Path to ImageNet pretrained CRATE weights.

domainnet_dataroot: Root path for DomainNet.

cubc_root: Root paths for CUB-C dataset.

### Scripts

Train the model:
```
bash scripts/domainnet.sh 0

bash scripts/ssbc.sh 0
```

## Acknowledgements

The codebase is largely built on this repo: https://github.com/CVMI-Lab/SimGCD.
