import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Convert color name to RGB vector
def color_to_rgb(color_name):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
    }
    return torch.tensor(colors[color_name.lower()], dtype=torch.float32) / 255.0

class PolygonColorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.inputs_dir = os.path.join(data_dir, "inputs")
        self.outputs_dir = os.path.join(data_dir, "outputs")
        self.json_path = os.path.join(data_dir, "data.json")
        self.transform = transform

        with open(self.json_path, 'r') as f:
            self.raw_data = json.load(f)
            self.mapping = {
                item["input_polygon"]: {"color": item["colour"], "output": item["output_image"]}
                for item in self.raw_data
            }
            self.image_names = list(self.mapping.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        item = self.image_names[idx]
        input_path = os.path.join(self.inputs_dir, item)
        output_path = os.path.join(self.outputs_dir, self.mapping[item]["output"])
        color_name = self.mapping[item]["color"]

        # Load images as grayscale and RGB
        input_img = Image.open(input_path).convert("L")   # 1 channel
        output_img = Image.open(output_path).convert("RGB")  # 3 channels

        # Resize if needed
        input_img = input_img.resize((128, 128))
        output_img = output_img.resize((128, 128))

        input_arr = np.array(input_img, dtype=np.float32) / 255.0
        output_arr = np.array(output_img, dtype=np.float32) / 255.0

        input_tensor = torch.tensor(input_arr).unsqueeze(0)  # Shape: [1, H, W]
        color_tensor = color_to_rgb(color_name).unsqueeze(1).unsqueeze(2)  # [3, 1, 1]
        color_tensor = color_tensor.expand(-1, 128, 128)  # [3, 128, 128]

        input_combined = torch.cat((input_tensor, color_tensor), dim=0)  # [4, H, W]
        output_tensor = torch.tensor(output_arr).permute(2, 0, 1)  # [3, H, W]

        return input_combined, output_tensor