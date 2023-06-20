from datetime import datetime
import os
import torch
from torch import nn
from data_loader import create_data_loader
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from utils import seeding, set_device
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# ----- set seed
seeding(42)

# ----- set device
device = set_device()

# ----- set data loader
data_dir = 'data/brain_tumor_dataset'
data_loaders = create_data_loader(path=data_dir, batch_size=64)
# train_loader = data_loaders['train']
# test_loader = data_loaders['test']

# ------ Model creation
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# ----- fine tuning for 2 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# save base model
# torch.save(model, 'saved_models/base_model.pt')

model.to(device)


# ----- create training loop
def train_model(model, data_loaders, optimizer, scheduler, device, criterion, tb_writer, num_epochs=10):
    loss_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        model.train()

        running_loss = 0.0
        # Iterate over data.
        for bi, d in enumerate(data_loaders["train"]):

            inputs = d[0]
            labels = d[1].unsqueeze(1)

            # convert labels to one-hot encoding
            # labels = torch.zeros(labels.size(0), 2).scatter_(1, labels, 1)

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            tb_writer.add_scalars("batch_loss", {"train": loss})
            tb_writer.add_scalars("batch_acc", {"train": torch.sigmoid(outputs).round().eq(labels).sum().item() / labels.size(0)})

        epoch_loss = running_loss / len(data_loaders["train"].dataset)
        loss_history.append(epoch_loss)

        # validation
        val_acc, val_loss = evaluate_model(model, data_loaders["val"], device, criterion)

        print('Train Loss: {:.4f}, Val Loss {:.4f}, Val Acc: {:.4f}'.format(epoch_loss, val_loss, val_acc))

        # push everything to tb
        tb_writer.add_scalars("loss", {"train": epoch_loss, "val": val_loss})
        tb_writer.add_scalars("acc", {"val": val_acc})

        scheduler.step()

    return model, loss_history


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    loss = []
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            inputs = d[0]
            labels = d[1].unsqueeze(1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            outputs = model(inputs)
            loss.append(criterion(outputs, labels))
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total, torch.stack(loss).mean()

# create log dir
log_dir = f"MRI_training_log/{datetime.now().strftime('%y-%m-%d_%H-%M')}"
os.makedirs(log_dir, exist_ok=True)

# ----- set input
optimizer_ft = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-6)
lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

criterion = nn.BCEWithLogitsLoss()
tb_writer = SummaryWriter(log_dir=log_dir)

model_ft, loss_history = train_model(model,
                       data_loaders,
                       optimizer_ft,
                       lr_sch,
                       criterion=criterion,
                       tb_writer=tb_writer,
                       num_epochs=20,
                       device=device)

# ----- evaluate model
acc, loss = evaluate_model(model_ft, data_loaders["test"], device, criterion=criterion)
print(f"Test Acc: {acc}, Test Loss: {loss}")

# ----- save model with time stamp
torch.save(model_ft.state_dict(), os.path.join(log_dir, f"model_state_dict.pt"))
torch.save(model_ft, os.path.join(log_dir, f"model.pt"))

# ----- save loss history
with open(os.path.join(log_dir, "loss_history.txt"), "w") as f:
    for loss in loss_history:
        f.write(f"{loss}\n")

# plot loss history
plt.plot(loss_history)
plt.title("Loss history")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 1)
# save the plot
plt.savefig(os.path.join(log_dir, "loss_history.png"))



