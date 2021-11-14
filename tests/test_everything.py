import pytest
import os
import requests
import torch
import numpy as np
from src.net import Net
from src.predict import predict


@pytest.fixture
def net():
    return Net()


def test_net(net):
    BATCH_SIZE = 2
    x = torch.zeros(BATCH_SIZE, 1, 28, 28)
    out = net(x)
    assert out.shape == (BATCH_SIZE, 10)


def test_predict():
    x = np.zeros((28, 28))
    checkpoint_path = "mnist_cnn.pt"
    if not os.path.exists(checkpoint_path):
        url = "https://github.com/stephenllh/mnist-mlops/releases/latest/download/mnist_cnn.pt"
        r = requests.get(url, allow_redirects=True)
        open(checkpoint_path, 'wb').write(r.content)
    log_preds = predict(x, checkpoint_path)
    assert log_preds.shape == (1, 10)
