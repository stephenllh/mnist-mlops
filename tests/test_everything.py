import pytest
import os
import requests
import torch
import numpy as np
from src.net import Net
from src.main import train, tst, get_loader
from src.predict import predict


@pytest.fixture
def net():
    return Net()


def test_net(net):
    BATCH_SIZE = 2
    x = torch.zeros(BATCH_SIZE, 1, 28, 28)
    out = net(x)
    assert out.shape == (BATCH_SIZE, 10)


class Args:
    dry_run = True
    log_interval = 100


def test_train_loop(net):
    device = "cpu"
    train_loader = get_loader(8, True)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=0.001)
    epoch = 1
    train(Args, net, device, train_loader, optimizer, epoch)


def test_testing_loop(net):
    device = "cpu"
    test_loader = get_loader(8, False)
    tst(net, device, test_loader)


def test_predict():
    x = np.zeros((28, 28))
    checkpoint_path = "mnist_cnn.pt"
    if not os.path.exists(checkpoint_path):
        url = "https://github.com/stephenllh/mnist-mlops/releases/latest/download/mnist_cnn.pt"
        r = requests.get(url, allow_redirects=True)
        open(checkpoint_path, 'wb').write(r.content)
    log_preds = predict(x, checkpoint_path)
    assert log_preds.shape == (1, 10)
