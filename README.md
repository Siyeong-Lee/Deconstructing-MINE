# Regularized MINE
## Environment
- Python 3.7.4
- pip-20.1.1
- Additional packages can be located in `requirements.txt` and `benchmark.requirements.txt` (Only for the `experiments_self_contrastive.py` and `experiments_classification.py`.

## How to reproduce experiments & figures:
### Figure 1. (a), (b), (c)
- observing_original_mine.ipynb
### Figure 2. (a), (b), (c), (d)
- observing_imine_in_original_mine_settings.ipynb
### Figure 3, 4, Table 1, 2
- Check with `python3 experiments_self_contrastive.py -h` or `python3 experiments_classification.py -h` for more details.
- Example. `python3 experiments_self_contrastive.py --dataset cifar100 --loss remine-0.01 --loss remine --batch_size 100 --model resnet18 --epochs 150 --seed 0`.
### Figure 5. (a), (b)
- observing_imine_different_target.ipynb
### Figure 5 (c)
- loss_surface_mine_vs_imine.ipynb
### Figure 6, Appendix Figure 7, 8, Table 3
- Run with `python3 experiments.py --batch_size 64 --loss {LOSS} --problem gaussian`
- `LOSS` can be one of `NWJ`, `iMINE` (ReMINE), `iMINE_j` (ReMINE-J), `SMILE`, `MINE`, `TUBA`, `InfoNCE`, `JS`, `TNCE`, `alpha`
- Visualize with comparison_with_sota.ipynb
- We modified code from https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
- We modified https://github.com/wgao9/mixed_KSG for nearest neighbor-based methods.
### Figure 9, 10
- Generate experiment script with `python3 generate_experiments_self_consistency.py > out.sh`
- `LOSS` can be one of `iMINE` (ReMINE), `iMINE_j` (ReMINE+J), `SMILE`, `SMILE_JS`, `MINE`, `InfoNCE`, `JS`, `alpha`
- We modified code from https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
