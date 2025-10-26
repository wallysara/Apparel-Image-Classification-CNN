# Clothing Image Classification using CNN (Keras/TensorFlow)

This project implements a **Convolutional Neural Network (CNN)** using **Keras/TensorFlow** to classify clothing images from the Fashion-MNIST dataset.  
It demonstrates data preprocessing, augmentation, model training, and full evaluation with visualization and confusion matrices.

---

## 🧠 Overview

- Built a deep CNN for multi-class clothing image classification.  
- Used **data augmentation** (rotation, zoom, shift) to improve generalization.  
- Performed **stratified train/validation split** for balanced training.  
- Trained the model with **Adam optimizer** and **categorical cross-entropy** loss.  
- Evaluated performance using accuracy metrics and **confusion matrices** for detailed analysis.

---

## ⚙️ Techniques Used

| Stage | Technique | Purpose |
|--------|------------|----------|
| Preprocessing | Normalization & One-Hot Encoding | Prepare data for CNN |
| Augmentation | Rotation, Zoom, Shift | Improve generalization |
| Model | Deep CNN with Dropout | Extract visual features |
| Evaluation | Accuracy & Confusion Matrix | Quantify and visualize performance |

---

## 🏗️ Model Architecture
Conv2D(32) → Conv2D(32) → MaxPool2D → Dropout(0.25)
Conv2D(64) → Conv2D(64) → MaxPool2D → Dropout(0.25)
Flatten → Dense(256, relu) → Dropout(0.5) → Dense(10, softmax


---

## 🚀 Training Configuration

- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Categorical Cross-Entropy  
- **Batch Size:** 300  
- **Epochs:** 50  
- **Dataset:** Fashion-MNIST (10 classes, 28×28 grayscale images)

---

## 🧪 Evaluation

- Achieved strong test accuracy after 50 epochs.  
- Visualized confusion matrices for train and test sets.  
- Analyzed per-class accuracy for deeper insights.

---

## 📊 Example Results

| Metric | Train | Validation | Test |
|---------|--------|-------------|------|
| Accuracy | ~98% | ~93% | ~91% |

Example Confusion Matrix visualization:  
✅ **T-shirt/top** ↔ Correctly classified  
⚠️ **Shirt** occasionally confused with **Coat**

---

## 📚 Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```
📖 References

Fashion-MNIST Dataset

Keras CNN Documentation

scikit-learn Confusion Matrix
