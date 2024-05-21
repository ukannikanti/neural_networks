import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Define the training data
input_size = 10
output_size = 1
num_samples = 1
X_train = torch.randn(num_samples, input_size)
y_train = torch.randn(num_samples, output_size)

# Instantiate the model
model = LinearModel(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
for param in model.parameters():
    print("==> ", param)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the trained model
with torch.no_grad():
    X_test = torch.randn(10, input_size)
    predicted = model(X_test)
    print("Predictions:", predicted)
