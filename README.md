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


Insights Report – Ayna ML Assignment
Hyperparameters Tried
Epochs: 20
Batch Size: 8
Learning Rate: 1e-4
Optimizer: Adam
Loss Function: L1Loss (better for smoother output than MSELoss)
Input Size: 128×128 (resized all images)

Model Design & Choices
Used a UNet architecture with skip connections and 4 encoder–decoder levels.
Input has 4 channels:
1 grayscale image (polygon shape)
3-channel RGB color condition
Output is 3-channel RGB colored polygon.
The model was trained from scratch, and no pretrained weights were used.

Training Trends
Trained over 20 epochs.

Validation loss decreased steadily, showing model generalization.

L1Loss provided visually smoother results, reducing color bleed or sharp edges.
Final Train Loss ≈ Lower than initial by over 40%
Final Validation Loss: consistent with training loss trend

Observed Failure Cases
Slight deviations from exact RGB shades in a few cases (e.g., cyan looked bluish).
Model performance improved with increased training epochs.

Key Learnings
Learned how to condition a UNet model on both image and color vectors.
Understood the impact of loss functions — L1Loss gave more visually coherent output than MSE.
Used Weights & Biases (wandb) to monitor training trends in real-time.
Built a clean training pipeline with reusable components (model, utils, dataset).

Future Improvements
Add data augmentation (rotation, noise) to improve generalization.
Try perceptual loss (like SSIM) for sharper and color-faithful outputs.
Move toward a conditional diffusion or attention-based architecture for complex variations.
