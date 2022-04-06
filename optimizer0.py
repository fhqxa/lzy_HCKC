from torch import optim


def Optimizer(switch_optimizer, model):
    if switch_optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=2e-4)
    elif switch_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
    elif switch_optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=1e-5)
    else:
        raise ValueError(switch_optimizer)

    return optimizer, model
