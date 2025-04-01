# 🧪 EDPO: Equivariant Denoising Diffusion Policy Optimization

Official codebase for the paper:  
**_Physics-Informed Policy Guided Diffusion for 3D Molecular Generation_**  

---

## 🧠 Overview

**EDPO** is a novel algorithm that fine-tunes *Equivariant Diffusion Models (EDMs)* for 3D molecular generation using **reinforcement learning**.  
It reframes the denoising process as a Markov Decision Process (MDP) and applies **policy optimization** using physicochemical reward signals such as:

- Molecular **stability**
- **Force-field** alignment

> 💡 EDPO significantly improves molecular generation stability and physical plausibility — crucial for tasks in drug discovery and material science.

---

## 📦 Installation

### 1. Setup Python Environment

```bash
conda create -n edpo python=3.10.14
conda activate edpo
pip install -r requirements.txt
```

### 2. Install xtb for Force Calculations (Used in Reward)

```bash
cd EDPO
git clone https://github.com/grimme-lab/xtb-python.git
cd xtb-python
pip install .
cd ..
```
## 🚀 Training & Fine-Tuning

### 🔧 Pretrain EDM Model on QM9

#### Pretrain EDM Model on QM9

```bash
python main_qm9.py \
  --n_epochs 3000 \
  --exp_name edm_qm9 \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 20 \
  --ema_decay 0.9999
```

#### 🔧 Pretrain EDM Model on GEOM-DRUGS

```bash
python main_geom_drugs.py \
  --n_epochs 3000 \
  --exp_name edm_geom_drugs \
  --n_stability_samples 500 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 4 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 1 \
  --ema_decay 0.9999 \
  --normalization_factor 1 \
  --model egnn_dynamics \
  --visualize_every_batch 10000
```

### 🧬 Fine-Tune with EDPO on QM9
QM-Force Reward
#### QM-Force Reward

```bash
python finetune.py \
  --config_path ./outputs/edm \
  --reward QM
```

#### GEN2-xTB Reward

```bash 
python finetune.py \
  --config_path ./outputs/edm \
  --reward xTB \
  --name edm_xTB
```

#### Stable Reward

```bash
python finetune.py \
  --config_path ./outputs/edm_qm9 \
  --reward Stable \
  --name edm_stable

```
### 🧬 Fine-Tune with EDPO on GEOM-DRUGS

#### GEN2-xTB Reward

```bash
python finetune.py \
  --config_path ./outputs/edm_geom_drugs \
  --reward xTB
 ```

### Finetune EDM Model with EDPO Algorithm on QM9 conditional on property.
#### GEN2-xTB
```python finetune.py --config_path ./outputs/edm_qm9 --reward xTB --name edm_xTB ```

## 📊 Evaluation
### Run analysis on a trained model:

```bash
python eval_analyze.py \
  --model_path ./exp/RewardQM \
  --n_samples 10000
```

