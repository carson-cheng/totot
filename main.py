import torchvision
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch
import torch.nn as nn
import os
def set_seeds(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
set_seeds(0)
from cox.utils import Parameters
import cox.store
def load_model(checkpoint_path="/workspace/resnet50_extra_pgd5_1epoch.pt", ds=None, batch_size=128, workers=8):
    if ds == None:
        ds = CIFAR('/path/to/cifar')
    m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=checkpoint_path)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path=checkpoint_path)
    return m, train_loader, val_loader
def train_model(m, train_loader, val_loader, config=None):
    if config == None:
        adv_train_kwargs = {
            'out_dir': "train_out",
            'epochs': 1,
            'lr': 0.002, # learning rate of the adversarial training
            'adv_train': 1,
            'adv_eval': 1,
            'constraint': 'inf', # attack norm
            'eps': 8/255, # attack epsilon
            'attack_lr': 2/255, # attack step size
            'attack_steps': 5 # number of steps
        } # 
    else:
        adv_train_kwargs = config
    
    adv_train_args = Parameters(adv_train_kwargs)
    os.system(f"mkdir {adv_train_kwargs['out_dir']}")
    # Fill whatever parameters are missing from the defaults
    adv_train_args = defaults.check_and_fill_args(adv_train_args,
                            defaults.TRAINING_ARGS, CIFAR)
    adv_train_args = defaults.check_and_fill_args(adv_train_args,
                            defaults.PGD_ARGS, CIFAR)

    # Train a model
    try:
        m = train.train_model(adv_train_args, m, (train_loader, val_loader))
    except AssertionError:
        m = train.train_model(adv_train_args, m.module, (train_loader, val_loader))
    return m
#m = train.train_model(adv_train_args, m.module, (train_loader, val_loader))
#train.train_model(adv_train_args, m, (train_loader, val_loader))
from robustness.train import eval_model
def eval_args(args, val_loader, m):
    adv_train_args = Parameters(args)

    # Fill whatever parameters are missing from the defaults
    adv_train_args = defaults.check_and_fill_args(adv_train_args,
                            defaults.TRAINING_ARGS, CIFAR)
    adv_train_args = defaults.check_and_fill_args(adv_train_args,
                            defaults.PGD_ARGS, CIFAR)
    try:
        eval_model(adv_train_args, m.module, val_loader, None)
    except AttributeError:
        eval_model(adv_train_args, m, val_loader, None)
#m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path="/workspace/cifar_l2_0_5.pt")
def evaluate(m, val_loader, norm):
    if norm not in ("inf", "2"):
        print("Invalid norm input")
        return
    adv_train_kwargs = {
        'out_dir': "train_out",
        'epochs': 1,
        'lr': 0.002, # at this checkpoint don't use an excessively high learning rate (0.001 is usually enough), or else it overfits to that threat model
        'adv_train': 1,
        'adv_eval': 1,
        'constraint': '2',
        'eps': 0.5,
        'attack_lr': 0.1,
        'attack_steps': 20
    }
    print("L2 PGD-20 evaluation:")
    eval_args(adv_train_kwargs, val_loader, m)    
    adv_train_kwargs = {
        'out_dir': "train_out",
        'epochs': 1,
        'lr': 0.002, # at this checkpoint don't use an excessively high learning rate (0.001 is usually enough), or else it overfits to that threat model
        'adv_train': 1,
        'adv_eval': 1,
        'constraint': '2',
        'eps': 0.5,
        'attack_lr': 0.1,
        'attack_steps': 100
    }
    if norm == "2":
        print("L2 PGD-100 evaluation:")
        eval_args(adv_train_kwargs, val_loader, m)
    adv_train_kwargs = {
        'out_dir': "train_out",
        'epochs': 1,
        'lr': 0.002, # at this checkpoint don't use an excessively high learning rate (0.001 is usually enough), or else it overfits to that threat model
        'adv_train': 1,
        'adv_eval': 1,
        'constraint': 'inf',
        'eps': 8/255,
        'attack_lr': 2/255,
        'attack_steps': 20
    }
    print("Linf PGD-20 evaluation:")
    eval_args(adv_train_kwargs, val_loader, m)
    adv_train_kwargs = {
        'out_dir': "train_out",
        'epochs': 1,
        'lr': 0.002, # at this checkpoint don't use an excessively high learning rate (0.001 is usually enough), or else it overfits to that threat model
        'adv_train': 1,
        'adv_eval': 1,
        'constraint': 'inf',
        'eps': 8/255,
        'attack_lr': 2/255,
        'attack_steps': 100
    }
    if norm == "inf":
        print("Linf PGD-100 evaluation:")
        eval_args(adv_train_kwargs, val_loader, m)
def main():
    m, train_loader, val_loader = load_model(checkpoint_path="/workspace/cifar_linf_8.pt")
    m = train_model(m, train_loader, val_loader)
    evaluate(m, val_loader, "inf")
if __name__ == "__main__":
    main()