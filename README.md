# Regularized MINE
## Environment
- Python 3.7.4
- pip-20.1.1
- Additional packages can be located in `requirements.txt`

## How to reproduce experiments & figures:
### Figure 1. (a), (b), (c), (d) Figure 2. (a), (b)
- observing_original_mine.ipynb
### Figure 3. (a), (b), (c), (d)
- observing_imine_in_original_mine_settings.ipynb
### Figure 4. (a), (b)
- observing_imine_different_target.ipynb
### Figure 5
- loss_surface_mine_vs_imine.ipynb
### Figure 6, Figure 10, Figure 11, Table 1
- Run with `python3 experiments.py --batch_size 64 --loss {LOSS} --problem gaussian`
- `LOSS` can be one of `NWJ`, `iMINE` (ReMINE), `iMINE_j` (ReMINE-J), `SMILE`, `MINE`, `TUBA`, `InfoNCE`, `JS`, `TNCE`, `alpha`
- Visualize with comparison_with_sota.ipynb
- We modified code from https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
- We modified https://github.com/wgao9/mixed_KSG for nearest neighbor-based methods.
### Figure 7, Figure 12, Figure 13, Figure 14
- Generate experiment sciript with `python3 generate_experiments_self_consistency.py > out.sh`
- `LOSS` can be one of `iMINE` (ReMINE), `iMINE_j` (ReMINE+J), `SMILE`, `SMILE_JS`, `MINE`, `InfoNCE`, `JS`, `alpha`
- We modified code from https://colab.research.google.com/github/google-research/google-research/blob/master/vbmi/vbmi_demo.ipynb
### Figure 8. (a), (b)
- observing_single_batch.ipynb
### Figure 8. (c)
- Run with `python3 run_single.py --gpu_id {GPU_ID} --batch_size {BATCH_SIZE} --regularizer_weight {LAMBDA}`
- Visualize with batchsize_experiment.ipynb
### Figure 9.
- observing_imine_continuous.ipynb
### Figure 15.
- We modified code from https://github.com/sudiptodip15/CCMI to test.
