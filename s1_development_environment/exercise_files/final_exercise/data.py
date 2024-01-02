import torch
import glob
import os

class CorruptedMNIST(torch.utils.data.Dataset):
    def __init__(self, folder, train = True):
        self.folder = folder
        # self.images_list = glob.glob(f'{folder}/{"train" if train else "test"}_images**.pt', recursive = True)
        # self.labels_list = glob.glob(os.path.join(folder, f'{"train" if train else "test"}_target*.pt'), recursive = True)
        if train:
            self.images_list = [os.path.join(folder, f'train_images_{i}.pt') for i in range(6)]
            self.labels_list = [os.path.join(folder, f'train_target_{i}.pt') for i in range(6)]
        else:
            self.images_list = [os.path.join(folder, f'test_images.pt')]
            self.labels_list = [os.path.join(folder, f'test_target.pt')]
        # print("Found {} images and {} labels".format(len(self.images_list), len(self.labels_list)))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        images = torch.load(self.images_list[idx])
        labels = torch.load(self.labels_list[idx])
        return images, labels

def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)
    # return train, test
    
    # mnist_path = os.path.join('data', 'corruptmnist')
    mnist_path = os.path.join('..', '..', '..', 'data', 'corruptmnist')
    
    train_set = CorruptedMNIST(mnist_path, train = True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = None) # already batched
    
    test_set = CorruptedMNIST(mnist_path, train = False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = None) # already batched
    
    return train_loader, test_loader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    train, test = mnist()
    for images, labels in train:
        print(images.shape)
        print(labels.shape)
        
        # display some examples
        fig, axes = plt.subplots(5, 5, figsize = (10, 10))
        for i in range(25):
            ax = axes.flatten()[i]
            ax.imshow(images[i].view(28, 28), cmap = 'gray')
            ax.set_axis_off()
            ax.set_title(labels[i])
        
        # plt.show()
            
        break
    for images, labels in test:
        print(images.shape)
        print(labels.shape)
        break
    # print(os.getcwd())
    # print(os.listdir('data'))