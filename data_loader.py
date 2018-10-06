import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt


def get_loader(name_dataset, batch_size, train=True):

    # Computed with compute_mean_std.py
    mean_std = {
        'amazon': {
            'mean': [0.79235494, 0.7862071 , 0.78418255],
            'std':  [0.31496558, 0.3174693 , 0.3193569 ]
        },
        'dslr': {
            'mean': [0.47086468, 0.44865608, 0.40637794],
            'std':  [0.20395322, 0.19204104, 0.1996422 ]
        },
        'webcam': {
            'mean': [0.6119875 , 0.6187739 , 0.61730677],
            'std':  [0.25063968, 0.25554898, 0.25773206]
        }
    }

    data_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[name_dataset]['mean'],
                                 std=mean_std[name_dataset]['std'])
        ])

    dataset = datasets.ImageFolder(root='./data/%s/images' % name_dataset,
                                   transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=train,
                                                 num_workers=4)
    return dataset_loader

