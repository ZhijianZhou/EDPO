# EDDPO
EDPO algorithm 
## Pretrain EDM Model on QM9
```python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999```
## Pretrain GEOM Model on GEOM
```python main_geom_drugs.py --n_epochs 3000 --exp_name edm_geom_drugs --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000```
## Finetune EDM Model with EDPO Algorithm on QM9
### QM-force
```python finetune.py --config_path ./outputs/edm_qm9 --reward QM ```
### GEN2-xTB
```python finetune.py --config_path ./outputs/edm_qm9 --reward xTB ```
### Stable
```python finetune.py --config_path ./outputs/edm_qm9 --reward Stable ```
## Finetune EDM Model with EDPO Algorithm on GEOM-DRUG
### GEN2-xTB
```python finetune.py --config_path ./outputs/edm_geom_drugs --reward xTB ```
## Analyze
```python eval_analyze.py --model_path ./exp/RewardQM --n_samples 10_000```
