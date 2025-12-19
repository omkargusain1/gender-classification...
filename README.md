# Gender Classification (Task A) - FACECOM Dataset

This project solves Task A of the Comsys Hackathon: Gender Classification using a custom CNN in Keras.

## 🧠 Approach

- Preprocess and resize all images to 224x224
- Use `ImageDataGenerator` for real-time data augmentation
- Train a simple yet effective CNN
- Evaluate using Accuracy, Precision, Recall, and F1-score

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── male/
│   └── female/
├── val/
│   ├── male/
│   └── female/
└── test/
    ├── male/
    └── female/
```

## 🚀 How to Run

1. **Train the Model**
```bash
python train.py
```

2. **Evaluate the Model**
```bash
python evaluate.py dataset/test
```

## 📝 Output

The model is saved as `gender_classification_model.h5`.  
Evaluation metrics include Accuracy, Precision, Recall, and F1-score.

## 📦 Dependencies

Install dependencies using:
```bash
pip install -r requirements.txt
```

## 🛠 Requirements

- TensorFlow
- scikit-learn
- numpy
