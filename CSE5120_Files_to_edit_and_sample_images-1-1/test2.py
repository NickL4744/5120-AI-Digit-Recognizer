# Step 1: Import all required keras libraries
from tensorflow import keras

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Step 2: Load and return training and test datasets
def load_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test

# Step 3: Define your CNN model
def digit_recognition_cnn():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Step 4: Call digit_recognition_cnn() to build your model
model = digit_recognition_cnn()

# Step 5: Train your model
X_train, X_test, y_train, y_test = load_dataset()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Step 6: Evaluate your model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Step 7: Save your model
model.save('digitRecognizer.h5')

# Step 8: Load and normalize new image
def load_new_image(path):
    new_image = load_img(path, grayscale=True, target_size=(28, 28))
    new_image = img_to_array(new_image)
    new_image = new_image.reshape((1, 28, 28, 1)).astype('float32')
    new_image = new_image / 255.0
    return new_image

# Step 9: Load a new image and predict its class
def test_model_performance():
    img = load_new_image("digit1")
    model = load_model('digitRecognizer.h5')
    imageClass = model.predict_classes(img)
    print("Predicted class:", imageClass[0])

# Step 10: Test model performance
test_model_performance()