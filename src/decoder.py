#
#       Simple AI-based image recognition script
#
#   Initialy written for https://doi.org/10.1016/j.photonics.2023.101222 
#
#   Modified for teaching students at the ITMO University
#
#   Serg S. Rzhevskii
#

# Connect the necessary libraries (mainly PyTorch)
import torch
import torch.nn as nn
from torch.autograd import Variable

# You need this to create your own dataset classes
import csv
from torch.utils.data import Dataset, DataLoader


# A class of training dataset. The peculiarity is that it contains correct labels (annotations) for neural network training.
# The training and test dataset are built similarly, I will describe this class in more detail
class MyTrainDataset(Dataset):
  # Class constructor
  def __init__(self):
    self.annotations = []
    with open("./train_dataset/dataset.csv") as fp:
      reader = csv.reader(fp, delimiter=",", quotechar='"')
      next(reader, None)  # skip the headers
      data_read = [row for row in reader]
      #Final annotations(labels)
      self.annotations = data_read

  #It needs to be able to return the size
  def __len__(self):
    return len(self.annotations)

  #We get the element
  def __getitem__(self, i):
    label = int(self.annotations[i][1])
    filepath = self.annotations[i][0]
    #Our map files 0 and 1 are in the data subfolder
    full_path = "./train_dataset/data/"+filepath
    data = []
    with open(full_path) as f:
      for line in f.readlines():
        #Remove 2 characters (line feed and \n).
        line = line[:-2]
        elements = line.split(" ")
        #Get an array of characters (so stored in CSV), convert to numbers
        for char in elements:
          if(char == "1"):
            data.append(1)
          else:
            data.append(0)

    #Return tensor(matrix) and label
    return torch.Tensor(data), label

# The class of the test dataset, i.e. on which we already determine
class MyTestDataset(Dataset):
  def __init__(self):
    self.annotations = []
    with open("./test_dataset/test_dataset.csv") as fp:
      reader = csv.reader(fp, delimiter=",", quotechar='"')
      next(reader, None)  # skip the headers
      data_read = [row for row in reader]
      self.annotations = data_read

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, i):
    #We don't need a label here
    #label = int(self.annotations[i][1])
    filepath = self.annotations[i][0]
    print(filepath)
    full_path = "./test_dataset/data/"+filepath
    data = []
    with open(full_path) as f:
      for line in f.readlines():
        #Remove 2 characters (line feed and \n).
        line = line[:-2]
        elements = line.split(" ")
        #Get an array of characters (so stored in CSV), convert to numbers
        for char in elements:
          if(char == "1"):
            data.append(1)
          else:
            data.append(0)

    #Now we return only the tensor
    return torch.Tensor(data)
    #return torch.Tensor(data), label


# Global settings of NN and workflow
# Sometimes these are called "hype parameters."
input_size = 36       # Image size = 6 x 6 = 36
hidden_size = 72      # Number of nodes on the hidden layer
num_classes = 3       # Number of output classes. In this case from 0 to 2 (total 3 pieces)
num_epochs = 10       # The amount of training of the entire dataset. It is important not to overtrain, otherwise you will overtrain!
batch_size = 9        # The size of input data for one iteration, it is useful to divide into "portions", batches
learning_rate = 0.01  # Learning rate. The bigger it is - the rougher, faster and worse the neuron learns

# Create datset instances
train_dataset = MyTrainDataset()
test_dataset = MyTestDataset()

# Create loaders - special instanses that properly prepare the dataset to work with PyTorch
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    shuffle=False
)

#NN class. Inherited from the base class from PyTorch
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()                    # Inherited by parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st linked layer: 36 (input data) -> 72 (hidden node)
        self.relu = nn.ReLU()                          # Nonlinear layer ReLU max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd linked layer: 72 (hidden node) -> 2 (output class)

    def forward(self, x):                              # i.e., prediction from entry to exit
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# It's all announced, so the work starts here.
net = Net(input_size, hidden_size, num_classes)

#If we want to enable the GPU, the cloud allows it, but we have it so easy we don't need it
#net.cuda()

# These complicated neural learning things
# The criterion for quality learning is the reduction of the so-called cross-entropy
criterion = nn.CrossEntropyLoss()
# Optimization variant - Adam's function
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# THERE IT IS! NN training!

# For each "epoch" of learning
for epoch in range(num_epochs):

    #For each item in the dataset
    for i, (data, labels) in enumerate(train_loader):

        #Uploading image data
        images = Variable(data)
        #Uploading lables
        labels = Variable(torch.Tensor(labels))

        optimizer.zero_grad()                             # Initialization of hidden weight to zeros
        outputs = net(images)                             # Feed forward: determining the output class, of a given image
        loss = criterion(outputs, labels)                 # Loss detection: difference between the output class and a predefined label
        loss.backward()                                   # refine weights
        optimizer.step()                                  # Optimizer: update weight parameters in hidden nodes

        #And this is where we report on the results
        if (i+1) % 1 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item()))


# At this stage we have a trained neuron, we will predict
print("Inference...")

#For each item in the test dataset
for data in test_loader:
    images = Variable(data)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Selecting the best class from the output: the class with the best score
    #Printing the answer. Format: Tensor[OUR PRESCRIBED CLASS]
    print("Predicted: ")
    print(predicted)