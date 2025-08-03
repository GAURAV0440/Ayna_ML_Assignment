# Ayna ML Assignment

This project is part of the Ayna ML Internship Assignment.  
It involves building a deep learning model (UNet) that takes a polygon image and a color name as input and outputs the same polygon filled with the specified color.

---

## Problem Statement

- **Input:**  
  1. Polygon image (e.g., triangle, square)  
  2. Color name (e.g., "red", "blue")

- **Output:**  
  RGB image of the polygon filled with the given color

---

## Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Weights & Biases (wandb)
- Jupyter Notebook

---
## 📁 Project Structure
<img width="459" height="398" alt="image" src="https://github.com/user-attachments/assets/b2a24978-ba6c-464a-9063-4fef995fdd7b" />


---

## How to Run the Project
### 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Download the dataset
Link: Download dataset (https://drive.google.com/file/d/1QXLgo3ZfQPorGwhYVmZUEWO_sU3i1pHM/view)
Extract it into the data/ folder

3. Train the model
python src/train.py
This trains the UNet model and saves it as models/unet_colored_polygon.pth

4. Test the model
jupyter notebook notebooks/inference.ipynb

This will show:
Input polygon
Ground truth
Predicted colored polygon

📌 Notes
.pth model file is ignored in Git due to size limits
The dataset is not included in the repo — download manually
The project works on both Google Colab and local Ubuntu setup


Insights Report
Hyperparameters Tried
Epochs: 20
Batch Size: 8
Learning Rate: 1e-4
Optimizer: Adam
Loss Function: L1Loss (gave smoother output than MSELoss)
Input Size: 128×128 (all images resized)

Model Design & Choices
Used a UNet architecture with 4 encoder–decoder levels and skip connections.
Input:
1-channel grayscale polygon image
3-channel RGB color tensor (as condition)
Combined into 4-channel input
Output: 3-channel RGB image
Trained from scratch, no pretrained weights used.

Training Trends
Model trained over 20 epochs.
Validation loss decreased steadily, showing good generalization.
L1Loss led to smoother color transitions and less edge artifacts.
Final Train Loss dropped by ~40%
Final Validation Loss followed the same downward trend

Observed Failure Cases
Slight mismatch in exact color tone (e.g., cyan appeared slightly bluish)
Performance improved with more training.

Key Learnings
Understood how to condition a UNet model using image + color input
Compared effect of loss functions (MSE vs L1)
Used Weights & Biases for tracking and logging
Built a modular PyTorch pipeline (model, dataloader, training, inference)

Future Improvements
Add data augmentation (e.g., rotation, scaling)
Try perceptual loss (SSIM) for better visual fidelity
Explore conditional diffusion or attention-based networks for more complex input-output control
