Image Classification CNN – Training & Deployment

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras for binary image classification.
The workflow includes:

Installing required dependencies

Training a CNN model (model.h5)

Using the trained model in a Streamlit web application (app.py)

Project Structure
project/
│
├── train_model.py        # Model training script
├── app.py                # Streamlit application
├── model.h5              # Saved trained model (generated after training)
├── requirements.txt      # Required Python libraries
│
└── TaskA.image/
    ├── train/
    │   ├── class_0/
    │   └── class_1/
    └── val/
        ├── class_0/
        └── class_1/


Note:
Folder names inside train and val represent class labels.

Requirements

Create a file named requirements.txt with the following content:

tensorflow
scikit-learn
numpy
streamlit
pillow

Step 1: Install Dependencies

It is recommended to use a virtual environment.

pip install -r requirements.txt

Step 2: Train the Model

Run the training script:

python train_model.py

What this script does:

Loads training and validation images using ImageDataGenerator

Trains a CNN model with multiple convolutional blocks

Applies:

Early stopping

Learning rate reduction

Best-model checkpointing

Evaluates performance using:

Accuracy

Precision

Recall

F1-score

Saves the trained model as:

model.h5

Step 3: Run the Streamlit Application

After training is complete and model.h5 is generated, run:

streamlit run app.py

Streamlit App Features:

Loads the trained model.h5

Accepts image input from the user

Preprocesses the image

Predicts and displays the class label (binary classification)

Model Output

Model file: model.h5

Loss function: Binary Crossentropy

Optimizer: RMSprop

Activation: Sigmoid (binary classification)

Notes & Best Practices

Ensure the dataset directory paths in train_model.py are correct for your system.

Always train the model before running the Streamlit app.

If predictions seem incorrect, retrain the model with more epochs or more data.

Troubleshooting

Issue: ValueError: input shape mismatch
Solution: Ensure images are RGB and resized to (200, 200).

Issue: model.h5 not found
Solution: Run train_model.py first to generate the model.

Author

Omkar Gusain
Department of Computer Science & Engineering
