# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries
import numpy as np
from keras.models import load_model # This is used to load your saved model
from keras.datasets import mnist # This is used to load mnist dataset later
from keras.utils import to_categorical # This will be used to convert your test image to a categorical class (digit from 0 to 9)
#changed np_utils to to_categorical due to an error that wouldn't allow the code to run

# Step 2: Load and return training and test datasets
def load_dataset():
    # 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 2b. reshape for X train and test vars - Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    
    # 2c. normalize inputs from 0-255 to 0-1 - Hint: X_train = X_train / 255
    X_train = X_train / 255
    X_test = X_test / 255
    
    # 2d. Convert y_train and y_test to categorical classes - Hint: y_train = to_categorical(y_train)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # 2e. return your X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

# Step 3: Load your saved model 
def load_saved_model():
    model = load_model('digitRecognizer.h5')
    return model

# Step 4: Evaluate your model via your_model_name.evaluate(X_test, y_test, verbose = 0) function
def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", score[0])
    print("Test Accuracy:", score[1])

# Code below to make a prediction for a new image.

# Step 5: This section below is optional and can be copied from your digitRecognizer.py file from Step 8 onwards - load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

 
# Step 6: load and normalize new image
def load_new_image(path):
    # 6a. load new image
    newImage = load_img(path, color_mode="grayscale", target_size=(28, 28))
    # 6b. Convert image to array
    newImage = img_to_array(newImage)
    # 6c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
    newImage = newImage.reshape(1, 28, 28, 1)
    # 6d. normalize image data - Hint: newImage = newImage / 255
    newImage = newImage / 255
    # 6e. return newImage
    return newImage
 
# Step 7: load a new image and predict its class
def test_model_performance(model):
    # 7a. Call the above load image function
    img = load_new_image('digit9.png')
    # 7b. predict the class probabilities
    class_probabilities = model.predict(img)
    # 7c. Extract the class with the highest probability
    predicted_class = np.argmax(class_probabilities, axis=1)
    # 7d. Print prediction result
    print("Predicted Class:", predicted_class[0])
 
# Step 8: Test model performance here by calling the above test_model_performance function
X_train, X_test, y_train, y_test = load_dataset()
model = load_saved_model()
evaluate_model(model, X_test, y_test)
test_model_performance(model)
