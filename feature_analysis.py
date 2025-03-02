#m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path="/workspace/cifar_linf_8.pt")
import torchvision
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torchattacks
import numpy as np
import torch.nn.functional as F
#from main import load_model
def analyze(m):
    device = torch.device('cuda')
    atk = torchattacks.PGD(nn.Sequential(m.normalizer, m.model).to(device))
    mean_output_change = 0.0
    distrib = []
    mean_sign_change = 0
    batch_count = 0
    for i, data in enumerate(val_loader, 0):
        test_tensor = data[0].to(device)
        model = nn.Sequential(m.normalizer, m.model).to(device)
        model[1].linear = nn.Identity()
        model.eval()
        output = model(test_tensor).detach()
        model.train()
        adv_tensor = atk(test_tensor, data[1].to(device))
        model = nn.Sequential(m.normalizer, m.model).to(device)
        model[1].linear = nn.Identity()
        #print(adv_tensor.shape)
        model.eval()
        adv_output = model(adv_tensor).detach()
        model.train()
        #print(adv_output.shape)
        #mean_output_change += torch.sum(torch.abs((adv_output - output) / output)).detach() # each feature changed by 0.0052
        pct_change = (torch.abs((adv_output - output) / output).cpu().detach())
        for item in pct_change:
            distrib.append(np.array(item))
        #distrib.append(pct_change)
        #mean_sign_change += (torch.sign(output) != torch.sign(adv_output)).detach().sum().item() # 9298 / (128 * 2048) features changed signs
        batch_count += 1
        if batch_count % 20 == 0:
            print(f"{batch_count} batches processed.")
    distrib = np.array(distrib).flatten()
    return distrib
m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path="/path/to/cifar_linf_8.pt")
baseline = analyze(m)
m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path="/path/to/resnet50_extra_pgd5_1epoch.pt")
pgd5 = analyze(m)
m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds, resume_path="/path/to/resnet50_extra_pgd40_1epoch.pt")
pgd40 = analyze(m)