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

## Project Structure
ayna-ml-assignment/
│
├── data/ # Dataset folder (training & validation)
├── models/ # Trained model (.pth file) [ignored in Git]
├── notebooks/
│ └── inference.ipynb # Testing and visualization notebook
├── src/
│ ├── model.py # UNet model definition
│ ├── train.py # Training script
│ └── utils.py # Custom dataset class & helper functions
│
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files (e.g., models/)
└── README.md # This file


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
