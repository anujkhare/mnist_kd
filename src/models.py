import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    @staticmethod
    def _make_encoder(n_in, n_out):
        return nn.Sequential(
            nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #             nn.BatchNorm2d(num_features=n_out),
        )

    def __init__(self ,) -> None:
        super().__init__()
        self.low_feat = self._make_encoder(1, 8)

        self.encoders = nn.Sequential(
            self._make_encoder(8, 16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self._make_encoder(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self._make_encoder(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            self._make_encoder(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.linears = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 512),
        )

        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        feats = self.low_feat(x)
        encoded = self.encoders(feats).view(-1, 512)
        preds = self.linears(encoded)
        preds = self.classifier(preds)
        return F.log_softmax(preds, dim=1)


def get_model_and_params(model_cls, gpu=None):
    # Test run!
    model = model_cls()

    img = torch.from_numpy(np.random.rand(1, 1, 32, 32).astype(np.float32))
    assert model(img).shape[1] == 10 and model(img).shape[0] == 1

    count = 0
    params = []
    for param in model.parameters():
        if param.requires_grad:
            count += np.prod(param.shape)
            params.append(param)

    print('{:,} trainable paramaters!'.format(count))
    if gpu is not None:
        model = model.cuda(device=gpu)

    return model, params