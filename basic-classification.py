"""
This part represents what I've learned from this tutorial :

* A neural network is basically a graph with layers

* A layer is a set of nodes called units
( There's many types of layers : Dense, Flatten, ... )

* A unit has a shape

* A shape is a tuple representing how many elements a tensor has in each dimension 
( a shape (30,4,10) means a tensor with 3 dimensions, containing 30 elements in the first dimension, 4 in the second and 10 in the third.  )

* A tensor is a structure that flows through the units of the NN 
( Usually compared to an N dimension array, but not an exact defenition )

for more infos : https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc#44748370

* An epoch is how many times ALL the dataset have been passed through the network

* A batch is a part of the data set that passes through the network at the same time

* the size of the data set / the size of the batch = number of iterations on the model to get 1 epoch

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import mnist_reader

# If using offline pre downloaded data 
"""
X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
"""

# If downloading the data, the data is stored at ~/.keras/datasets
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
mnist_reader.valid_imshow_data(train_images[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# To normalize the colors ( make it between 0 and 1 ) 
train_images = train_images / 255.0
test_images = test_images / 255.0

# Fixing the size of the figure by 10x10 (inch²) 
plt.figure(figsize=(10,10))
# Example to see 25 samples from the trainig dataset
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.get_cmap('binary'))
    plt.xlabel(class_names[train_labels[i]])

# Creating a Neural Network with 3 layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Train the model with the AdamOptmizer algorithm ( I read its paper )
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 5 epochs means that the ALL data is passed through the network 5 times
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Predict from the test dataset using the trained model
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

# Define a function to plot the images
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.get_cmap('binary'))

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 6
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)


# Grab an image from the test dataset
img = test_images[0]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


np.argmax(predictions_single[0])

plt.show()

