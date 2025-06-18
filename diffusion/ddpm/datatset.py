import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Lambda, Compose

class DDPMDataset(Dataset):
    """
    A dataset class for loading and preprocessing images for DDPM training.
    This class supports loading datasets like MNIST and CIFAR10.
    """

    def __init__(self, dataset_name='MNIST'):
        self.dataset_name = dataset_name
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda x: (x - 0.5) * 2) # Normalize to [-1, 1]
        ])
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == 'MNIST':
            return torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        elif self.dataset_name == 'CIFAR10':
            return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label