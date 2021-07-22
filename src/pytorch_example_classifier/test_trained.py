import torch
import torchvision
from src.pytorch_example_classifier.download_data import load_cifar
from src.pytorch_example_classifier.save_images_test import save_img
from src.pytorch_example_classifier.nn import Net

testLoader = load_cifar(2)
dataiter = iter(testLoader) #2 = testLoader
images, labels = dataiter.next()

#test img
classes = load_cifar(3)
save_img.imshow(save_img, torchvision.utils.make_grid(images))
print("Test image classes: ", " ".join('%5s' % classes[labels[j]] for j in range(4)))

#load saved model
net = Net()
PATH = './src/pytorch_example_classifier/cifar_net.pth'
net.load_state_dict(torch.load(PATH))

#run test img through net
outputs = net(images) #outputs = eneriges for all classes
#printing the class with the highest energy
_, predicted = torch.max(outputs, 1)
print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# How does the network perform on the whole dataset:
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))