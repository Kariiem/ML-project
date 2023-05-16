import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import utils
from sklearn.metrics import f1_score

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, dropout=0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(17, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 15)
        self.fc5 = nn.Linear(15, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x= self.dropout(x)
        x = torch.relu(self.fc2(x))
        x= self.dropout(x)
        x = torch.relu(self.fc3(x))
        # x= self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        # x = torch.softmax(x, dim=1)  # Apply the softmax function along the class dimension

        return x

def csv_to_tensor(df):
    # convert csv to tensor
    X = df.agg(utils.transform_dataset) # utils.transform_dataset is a dicitionary which applies a transforming function on each column
    y= X["Body_Level"]
    # y= y-1
    X = X.drop("Body_Level", axis=1)
    X["BMI"] = X["Weight"].astype(float) / (X["Height"] ** 2).astype(float)
    X_tensor = torch.from_numpy(X.values).float()
    y_tensor = torch.from_numpy(y.values).long()
    return X_tensor, y_tensor

# if main
if __name__ == '__main__':
    net = Net()
    # load model
    net.load_state_dict(torch.load('ffn_model.pt'))

    # get csv name
    input_csv = input("Enter the name of the csv file: ")
    # load csv
    df = pd.read_csv(input_csv)
    # convert csv to tensor
    X_tensor, y_tensor = csv_to_tensor(df)
    # predict
    with torch.no_grad():
        output = net(X_tensor)
        _, predicted = torch.max(output.data, 1)
    # print acc
    print("Accuracy: ", (predicted == y_tensor).sum().item() / len(y_tensor))
    # print F1
    print('F1 score of the network on the test data: %.3f' % (f1_score(y_tensor, predicted, average='macro')))

    # loop over X_tensor and print answers one by one
    y_numpy = y_tensor.numpy()
    true_count = 0
    for i in range(len(X_tensor)):
        with torch.no_grad():
            output = net(X_tensor[i])
            _, predicted = torch.max(output.data, 0)
        # print("The predicted body level is: ", predicted.item()+1)
        predicted_numpy = predicted.numpy()
        if (i==1000):
            print("The predicted body level is: ", predicted_numpy)
            print("The actual body level is: ", y_numpy[i])







