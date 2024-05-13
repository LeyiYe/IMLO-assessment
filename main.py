from torchvision import datasets
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.nn as nn
import torch

training_data1 = datasets.Flower102(
    root="data",
    split="train",
    download=True,
    transform=ToTensor()

)

validation_data1 = datasets.Flowers102(
    root="data",
    split="train",
    download=True,
    transform=ToTensor()
)

training_data = Subset(training_data1, range(200))
validation_data = Subset(validation_data1, range(200))

training_images = [image for image, label in training_data]
training_labels = [label for image, label in training_data]

test_images = [image for image, label in validation_data]
test_labels = [label for image, label in validation_data]

max_width = 0
max_height = 0
for i in range(0, 100, 1):
    if training_images[i].shape[1] > max_width:
        max_width = training_images[i].shape[1]
    if training_images[i].shape[2] > max_height:
        max_height = training_images[i].shape[2]
    if test_images[i].shape[1] > max_width:
        max_width = test_images[i].shape[1]
    if test_images[i].shape[2] > max_height:
        max_height = test_images[i].shape[2]

resized_training_images = []
for image in training_images:
    resized_image = F.interpolate(image.unsqueeze(0), size=(max_height, max_width), mode='bilinear',
                                  align_corners=False)
    resized_training_images.append(resized_image.squeeze(0))

resized_training_images = tuple(resized_training_images)

resized_testing_images = []
for image in test_images:
    resized_image = F.interpolate(image.unsqueeze(0), size=(max_height, max_width), mode='bilinear',
                                  align_corners=False)
    resized_testing_images.append(resized_image.squeeze(0))

resized_testing_images = tuple(resized_testing_images)


class My_NN(nn.Module):
    def __init__(self, in_feature, hidden_layers, out_features, activation_function):
        super().__init__()
        if len(hidden_layers) < 1:
            raise Exception("My_NN must at least have 1 hidden layer")
        self.layers = []
        self.layers.append(nn.Linear(in_feature, hidden_layers[0]))
        self.add_module("input_layer", self.layers[0])

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.add_module(f"hidden_layer_{i}", self.layers[i])

        self.out = nn.Linear(hidden_layers[-1], out_features)

        self.activation_function = activation_function

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation_function(self.layers[i](x))
        x = self.out(x)
        return x


image_size = max_width
classifier = My_NN(in_feature=image_size, hidden_layers=[16,8], out_features=102, activation_function=F.relu)

resized_training_images = torch.stack(resized_training_images)
resized_testing_images = torch.stack(resized_testing_images)

training_labels = torch.LongTensor(training_labels).unsqueeze(1)
test_labels = torch.LongTensor(test_labels).unsqueeze(1)
training_labels = torch.LongTensor(training_labels).unsqueeze(1)
test_labels = torch.LongTensor(test_labels).unsqueeze(1)

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
epochs = 100

losses = []
for i in range(epochs):
    y_pred = classifier.forward(resized_training_images)

    loss = lossFn(y_pred, training_labels.unsqueeze(1))

    if i % 10 == 0:
        print(f"Epoch {i} - {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
