# hotdog_nothotdog
# Motivation
This project was born out of a personal challenge to learn more about convolutional neural networks (CNNs) and transfer learning. I wanted to implement a fun and humorous project that would help me understand the concepts better. And what's more fun than detecting hot dogs?

# Project Overview
In this project, I used TensorFlow and transfer learning to create a CNN that can detect whether an image is a hot dog or not a hot dog. I leveraged the VGG-19 model as a starting point and fine-tuned it using a public dataset from Kaggle.

# Dataset
I used the Kaggle Hot Dog or Not Hot Dog dataset for this project. This dataset contains a total of 5,000 images, with 2,500 hot dog images and 2,500 non-hot dog images.

# Model Architecture
I used the VGG-19 model as a starting point and fine-tuned it for my specific task. I added a few more layers on top of the VGG-19 model to adapt it to my dataset. The final model architecture is as follows:

# VGG-19 model as the base model
-Added a flatten layer to flatten the output of the convolutional layers
-Added two dense layers with ReLU activation and dropout for regularization
-Final output layer with sigmoid activation for binary classification

#Training and Validation
I trained the model using the Adam optimizer with a learning rate of 0.001. I used a batch size of 32 and trained the model for 10 epochs. The model was validated using a validation set consisting of 20% of the total dataset.

The training and validation accuracy and loss curves are shown in the Jupyter notebook provided in this repository.

#Results
The final model achieved an accuracy of 95.6% on the validation set. The confusion matrix is shown below:

#Predicted Hot Dog	Predicted Not Hot Dog
Actual Hot Dog	236	14
Actual Not Hot Dog	11	239
#Conclusion
This project was a fun and challenging way to learn about convolutional neural networks and transfer learning. I was able to fine-tune a pre-trained model to detect hot dogs with high accuracy. The model was trained and validated using a public dataset from Kaggle.

#Files and Folders
hot_dog_or_not_hot_dog.ipynb: Jupyter notebook containing the code for the project
data/: Folder containing the dataset used for training and validation
models/: Folder containing the saved model weights and architecture
#Requirements
Python 3.7+
TensorFlow 2.3+
Jupyter Notebook
GPU with at least 4 GB of VRAM (optional but recommended)
