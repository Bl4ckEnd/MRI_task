import torch
from torch import nn
from data_loader import create_data_loader
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from utils import set_device
import matplotlib.pyplot as plt


# ----- set device
device = set_device()

# ----- set data loader
data_dir = 'data/brain_tumor_dataset'
data_loaders = create_data_loader(path=data_dir, batch_size=64)
train_loader = data_loaders['train']
test_loader = data_loaders['test']

# ------ Model creation
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# ----- fine tuning for 2 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# save base model
# torch.save(model, 'saved_models/base_model.pt')

model.to(device)


# ----- create training loop
def train_model(model, data_loader, optimizer, scheduler, device, num_epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    loss_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        scheduler.step()
        model.train()

        running_loss = 0.0
        # Iterate over data.
        for bi, d in enumerate(data_loader):

            inputs = d[0]
            labels = d[1].unsqueeze(1)

            # convert labels to one-hot encoding
            labels = torch.zeros(labels.size(0), 2).scatter_(1, labels, 1)

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(data_loader.dataset)
        print('Loss: {:.4f}'.format(epoch_loss))
        loss_history.append(epoch_loss)
    return model, loss_history


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            inputs = d[0]
            labels = d[1].unsqueeze(1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# ----- set input
optimizer_ft = optim.Adam(params=model.parameters(), lr=0.001)
lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model_ft, loss_history = train_model(model,
                       train_loader,
                       optimizer_ft,
                       lr_sch,
                       num_epochs=20,
                       device=device)

# ----- save model with time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")
torch.save(model_ft.state_dict(), f"saved_models/{timestr}_model.pt")

# ----- save loss history
with open(f"saved_models/{timestr}_loss_history.txt", "w") as f:
    for loss in loss_history:
        f.write(f"{loss}\n")

# plot loss history
plt.plot(loss_history)
plt.title("Loss history")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.show()

# ----- evaluate model
acc = evaluate_model(model_ft, test_loader, device)
print(f"Accuracy: {acc}")


