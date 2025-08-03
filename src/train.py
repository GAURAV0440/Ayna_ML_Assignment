import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.model import UNet
from src.utils import PolygonColorDataset
import wandb
from tqdm import tqdm

# --- Config ---
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- wandb Init ---
wandb.init(
    entity="gchawla365-gauravchawla111",  # ✅ Your correct team name
    project="ayanaproject",               # ✅ Your exact wandb project name
    name="unet-train",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "loss_function": "L1Loss",
    }
)

# --- Datasets & Loaders ---
train_dataset = PolygonColorDataset("data/dataset/training")
val_dataset = PolygonColorDataset("data/dataset/validation")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --- Model, Loss, Optimizer ---
model = UNet().to(DEVICE)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_output = model(x)
            total_val_loss += criterion(val_output, y).item()

    avg_val_loss = total_val_loss / len(val_loader)

    # --- wandb Log ---
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/unet_colored_polygon.pth")
print("✅ Model saved to models/unet_colored_polygon.pth")