import torch
import torch.nn as tn
import torch.optim as optim
from src.pytorch_example_classifier.download_data import load_cifar
from src.pytorch_example_classifier.nn import Net


##Define a Loss function and optimizer
net = Net()
PATH = './src/pytorch_example_classifier/cifar_net.pth'
net.load_state_dict(torch.load(PATH))
criterion = tn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(load_cifar(1), 0): #1 = trainLoader
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

##Saving the trained model
PATH = './src/pytorch_example_classifier/cifar_net.pth'
torch.save(net.state_dict(), PATH)