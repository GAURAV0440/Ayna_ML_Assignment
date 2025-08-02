import sys
import os
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

# --- Initialize Weights & Biases ---
wandb.init(project="ayna-ml-assignment", name="unet-train", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LEARNING_RATE,
    "loss": "L1Loss"
})

# --- Dataset ---
train_dataset = PolygonColorDataset("data/dataset/training")
val_dataset = PolygonColorDataset("data/dataset/validation")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = UNet().to(DEVICE)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    val_loss /= len(val_loader)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_loss": val_loss
    })

    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/unet_colored_polygon.pth")
print("✅ Model saved to models/unet_colored_polygon.pth")