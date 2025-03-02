with open("/usr/local/lib/python3.10/dist-packages/robustness/imagenet_models/__init__.py") as rf:
    contents = rf.read()
    contents = contents.replace("from .vgg import *", "#")
    contents = contents.replace("from .alexnet import *", "#")
    contents = contents.replace("from .squeezenet import *", "#")
    with open("/usr/local/lib/python3.10/dist-packages/robustness/imagenet_models/__init__.py", "w") as wf:
        wf.write(contents)

with open("/usr/local/lib/python3.10/dist-packages/robustness/train.py") as rf:
    contents = rf.read()
    target_string = '''    for i, (inp, target) in iterator:
       # measure data loading time
        target = target.cuda(non_blocking=True)
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)'''
    replace_string = '''    import random
    import copy
    for i, (inp, target) in iterator:
       # measure data loading time
        target = target.cuda(non_blocking=True)
        atk_kwargs = {}
        atk = random.randint(1, 3)
        if atk == 1:
            atk_kwargs = copy.deepcopy(attack_kwargs)
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)'''
    contents = contents.replace(target_string, replace_string)
    with open("/usr/local/lib/python3.10/dist-packages/robustness/train.py", "w") as wf:
        wf.write(contents)