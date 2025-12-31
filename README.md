Image Classification CNN – Training & Deployment

A complete end-to-end deep learning pipeline for binary image classification using TensorFlow/Keras, with an interactive Streamlit web application for real-time predictions.

This project is designed to be easy to run, easy to understand, and easy to extend.

🚀 Project Workflow

Install required dependencies

Train a CNN model and generate model.h5

Deploy the trained model using a Streamlit web app

📁 Project Structure
project/
│
├── train_model.py        # CNN training script
├── app.py                # Streamlit web application
├── model.h5              # Trained model (generated after training)
├── requirements.txt      # Python dependencies
│
└── TaskA.image/
    ├── train/
    │   ├── class_0/
    │   └── class_1/
    └── val/
        ├── class_0/
        └── class_1/

ℹ️ Important Note

Folder names inside train/ and val/ act as class labels

Images must be RGB and will be resized to 200 × 200

🛠️ Requirements

Create a file named requirements.txt with the following content:

tensorflow
scikit-learn
numpy
streamlit
pillow

⚙️ Step 1: Install Dependencies

It is highly recommended to use a virtual environment.

pip install -r requirements.txt

🧠 Step 2: Train the Model

Run the training script:

python train_model.py

🔍 What Happens During Training?

Loads images using ImageDataGenerator

Applies image rescaling and shuffling

Trains a deep CNN with multiple convolution blocks

Uses advanced training callbacks:

Early Stopping – prevents overfitting

Reduce Learning Rate on Plateau

Best Model Checkpointing

Evaluates the model using:

Accuracy

Precision

Recall

F1-Score

Saves the best performing model as:

model.h5

🌐 Step 3: Run the Streamlit Web App

Once model.h5 is generated, launch the app:

streamlit run app.py

✨ Streamlit App Features

Loads the trained CNN model

Allows users to upload images

Automatically preprocesses the input

Performs binary classification

Displays prediction results instantly

📊 Model Details
Component	Description
Model Type	Convolutional Neural Network (CNN)
Input Size	200 × 200 × 3
Loss Function	Binary Crossentropy
Optimizer	RMSprop
Output Activation	Sigmoid
Output	Binary Class Prediction
✅ Best Practices

Always train the model first before running the Streamlit app

Verify dataset paths inside train_model.py

Use more data and epochs for better accuracy

Maintain class balance for stable predictions

🐞 Troubleshooting
❌ Error: ValueError: input shape mismatch

✔ Solution:
Ensure all images are RGB and resized to 200 × 200

❌ Error: model.h5 not found

✔ Solution:
Run train_model.py to generate the trained model

📌 Future Enhancements (Optional)

Transfer learning (VGG16, MobileNet, ResNet)

Multi-class classification

Model performance visualization

Docker deployment

👨‍💻 Author

Omkar Gusain
Department of Computer Science & Engineering
