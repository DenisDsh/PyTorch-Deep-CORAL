import torch
from torchvision import transforms, datasets
import numpy as np


def compute_mean_std(path_dataset):
    """
    Compute mean and standard deviation of an image dataset.
    Acknowledgment : http://forums.fast.ai/t/image-normalization-in-pytorch/7534
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=path_dataset,
                                   transform=transform)
    # Choose a large batch size to better approximate. Optimally load the dataset entirely on memory.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)

    pop_mean = []
    pop_std = []

    for i, data in enumerate(data_loader, 0):
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,) -> 3 channels
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))

        pop_mean.append(batch_mean)
        pop_std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)

    values = {
        'mean': pop_mean,
        'std': pop_std
    }

    return values


def main():
    mean_std = {}
    for dataset in ['amazon', 'dslr', 'webcam']:
        # Construct path
        dataset_path = './data/%s/images' % dataset
        values = compute_mean_std(dataset_path)
        # Add values to dict
        mean_std[dataset] = values

    print(mean_std)

if __name__ == '__main__':
    main()