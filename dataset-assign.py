import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 784
hidden_size = 128
output_size = 10


model = SimpleNN(input_size, hidden_size, output_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
batch_size = 64
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:

        optimizer.zero_grad()


        inputs = inputs.view(-1, input_size)


        outputs = model(inputs)


        loss = criterion(outputs, labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

print("Training complete")


torch.save(model.state_dict(), 'simple_nn_model.pth')
