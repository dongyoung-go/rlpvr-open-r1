import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()
accelerator.print(f"Process {accelerator.process_index}: Accelerator initialized!")

# Create dummy dataset
x = torch.randn(1000, 10)  # 1000 samples, 10 features
y = torch.randint(0, 2, (1000,))  # Binary classification labels

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

accelerator.print(f"Process {accelerator.process_index}: Dataset and DataLoader created!")

# Define simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Prepare model, optimizer, and data for distributed training
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

accelerator.print(f"Process {accelerator.process_index}: Model, Optimizer, and DataLoader prepared!")

# Training loop
for epoch in range(3):
    model.train()
    total_loss = 0
    accelerator.print(f"Process {accelerator.process_index}: Starting epoch {epoch+1}")

    for step, batch in enumerate(dataloader):
        x_batch, y_batch = batch

        accelerator.print(f"Process {accelerator.process_index}: Forward pass at step {step}")
        optimizer.zero_grad()
        outputs = model(x_batch)

        accelerator.print(f"Process {accelerator.process_index}: Computing loss at step {step}")
        loss = loss_fn(outputs, y_batch)

        accelerator.print(f"Process {accelerator.process_index}: Backward pass at step {step}")
        accelerator.backward(loss)

        accelerator.print(f"Process {accelerator.process_index}: Optimizer step at step {step}")
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            accelerator.print(f"Process {accelerator.process_index}: Step {step}, Loss: {loss.item()}")

    accelerator.print(f"Process {accelerator.process_index}: Epoch {epoch+1} completed, Avg Loss: {total_loss / len(dataloader)}")

print("Training finished!")
