import torch.utils.data
import torchvision
from torchvision import transforms


def main():
    dataset = torchvision.datasets.CIFAR100("./cifar-100", download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for images, labels in loader:
        print(batch.shape)


if __name__ == '__main__':
    main()