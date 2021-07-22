import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar(input_int : int):
    #The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainSet = torchvision.datasets.CIFAR10(root = "/workspaces/gerbert-punctuator/src/pytorch_example_classifier/data", train = True, download = False, transform = transform)
    #num_workers: each worker loads a single batch and returns it only once it’s ready.
    trainLoader = DataLoader(trainSet, batch_size = batch_size, shuffle = True, num_workers = 2)
    testSet = torchvision.datasets.CIFAR10(root = "/workspaces/gerbert-punctuator/src/pytorch_example_classifier/data", train = False, download = False, transform = transform)
    testLoader = DataLoader(testSet, batch_size = batch_size, shuffle = False, num_workers = 2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if input_int == 1:
        return trainLoader
    elif input_int == 2:
        return testLoader
    elif input_int == 3:
        return classes
    elif input_int == 4:
        return batch_size