import torchvision
import matplotlib.pyplot as plt
import numpy as np
from src.pytorch_example_classifier.download_data import load_cifar

class save_img:
    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("./src/pytorch_example_classifier/test.png")

    def save_test_image(self):
        # get some random training images
        dataiter = iter(load_cifar(1)) #trainloader
        images, labels = dataiter.next()
        # save images
        self.imshow(self, torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % load_cifar(3)[labels[j]] for j in range(load_cifar(4)))) # 3 = classes, 4 = batch_size
    
save_img.save_test_image(save_img)