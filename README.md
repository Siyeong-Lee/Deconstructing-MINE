# Interpretable MINE
## Environment
- Python 3.7.4
- pip-20.1.1
- Additional packages can be located in `requirements.txt`

## How to reproduce experiments & figures:
### Figure 1. (a), (b), Figure 2. (a), (b)
- observing_original_mine.ipynb
### Figure 3. (a), (b)
- observing_imine_different_target.ipynb
### Figure 4
- loss_surface_mine_vs_imine.ipynb
### Figure 5. (a), (b)
- observing_single_batch.ipynb
### Figure 5. (c)
- Run with `python3 run_single.py --gpu_id {GPU_ID} --batch_size {BATCH_SIZE} --regularizer_weight {LAMBDA}`
- Visualize with batchsize_experiment.ipynb
### Figure 6. (a), (b)
- observing_imine_continuous.ipynb
### Figure 7, Appendix 3:
- Run with `python3 experiments.py --batch_size 64 --loss {LOSS} --problem gaussian`
- `LOSS` can be one of `NWJ`, `iMINE`, `iMINE_j`, `SMILE`, `MINE`, `TUBA`, `InfoNCE`, `JS`, `TNCE`, `alpha`
- Visualize with comparison_with_sota.ipynb
