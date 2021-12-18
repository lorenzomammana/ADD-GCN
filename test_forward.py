import ssl

import torch
import torchvision

from models.add_gcn import ADD_GCN

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    res101 = torchvision.models.resnet101(pretrained=True)
    model = ADD_GCN(res101, 10)
    model.eval()

    input_tensor = torch.zeros([1, 3, 448, 448])

    model.forward(input_tensor)
