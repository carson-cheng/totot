# Training on Top of Training: Improving Adversarially Trained Model Checkpoints

Code for a very small (for now) empirical analysis on how to improve adversarially trained model checkpoints by training on different attack hyperparameters

To experiment with the code:
1. Download the model checkpoints from https://github.com/MadryLab/robustness/tree/master.
2. Run `main.py`
3. Look for a `train_out` directory. Select the model based on `checkpoint_best.pt` (if you don't mind rerunning the evaluation already in `main.py`) or `checkpoint_latest.pt` (if you want the model that has already been evaluated)

To download the model checkpoints:
Download the model checkpoints from https://drive.google.com/drive/folders/1zkFSW4mfScVUycWqBljEEkS6KToldXSB?usp=sharing

Checkpoint accuracies:
|                                                     | Natural | PGD-20 (Linf) | PGD-100 (Linf) | PGD-20 (L2) |
|-----------------------------------------------------|---------|---------------|----------------|-------------|
| Baseline                                            | 87.03   | 52.87         | 52.33          | 28.31       |
| Natural, 1 epoch (resnet50_extra_natural_1epoch.pt) | 91.34   | 34.88         | 33.95          | 15.46       |
| PGD-1, 1 epoch (resnet50_extra_pgd1_1epoch.pt)      | 90.51   | 44.40         | 43.54          | 21.08       |
| PGD-5, 1 epoch (resnet50_extra_pgd5_1epoch.pt)      | 87.19   | 53.44         | 52.88          | 28.68       |
| PGD-10, 1 epoch (resnet50_extra_pgd10_1epoch.pt)    | 87.02   | 53.70         | 53.31          | 29.14       |
| PGD-40, 1 epoch (resnet50_extra_pgd40_1epoch.pt)    | 86.80   | 54.54         | 54.15          | 29.97       |

|                                                                     | Natural | PGD-20 (Linf) | PGD-100 (Linf) |
|---------------------------------------------------------------------|---------|---------------|----------------|
| PGD-5, 1 epoch, step size 1/255 (resnet50_extra_pgd5_1_1epoch.pt)   | 89.18   | 49.49         | 48.97          |
| PGD-5, 1 epoch, step size 4/255 (resnet50_extra_pgd5_4_1epoch.pt)   | 87.30   | 53.53         | 53.21          |
| PGD-10, 1 epoch, step size 1/255 (resnet50_extra_pgd10_1_1epoch.pt) | 87.49   | 53.30         | 52.80          |
| PGD-10, 1 epoch, step size 4/255 (resnet50_extra_pgd10_4_1epoch.pt) | 87.04   | 53.86         | 53.48          |

|                                                     | Natural | PGD-20 (L2) | PGD-100 (L2) | PGD-20 (Linf) |
|-----------------------------------------------------|---------|-------------|--------------|---------------|
| Baseline                                            | 90.83   | 69.93       | 69.65        | 31.66         |
| PGD-5, 1 epoch (resnet50_extra_pgd5_l2_1epoch.pt)   | 91.06   | 69.91       | 69.71        | 30.85         |
| PGD-10, 1 epoch (resnet50_extra_pgd10_l2_1epoch.pt) | 91.07   | 70.29       | 70.03        | 31.36         |
| PGD-40, 1 epoch (resnet50_extra_pgd40_l2_1epoch.pt) | 91.03   | 70.31       | 70.04        | 31.47         |
