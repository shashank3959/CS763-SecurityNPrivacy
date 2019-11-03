import tensorflow as tf

BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

ENSEMBLE_SIZE = 10
PRUNE_FACTOR = 0.1

# Enable eager so that dataset iterator will work
tf.enable_eager_execution()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
input_shape = (32, 32, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(TEST_BATCH_SIZE)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


model = tf.keras.models.load_model('cifar10_new.h5')

scc = tf.keras.losses.SparseCategoricalCrossentropy()


def loss(model, x, y):
    y_ = model(x)
    return scc(y, y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


#features, labels = next(iter(train_dataset))

#print(grad(model, features, labels) )

from tensorflow import contrib
tfe = contrib.eager

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)
num_epochs = 1

train_loss_results = []
train_accuracy_results = []

i = 0

LAYER_TUPLES = None
NUM_LAYERS = None









import numpy as np
from scipy import stats
model_copy= tf.keras.models.clone_model(model)
import itertools
import random




perturbed_accuracy = tfe.metrics.Accuracy()
defense_accuracy = tfe.metrics.Accuracy()


num = 0
ec = 0
ea = 0


i = 0
for (x, y) in test_dataset:

    i += 1
    if i == 100:
        break;
    y__ = model(x)

    classes = []
    for j in range(11):

        y__ = tf.argmax(y__, axis=1)

        classes.append(y__)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss2 = loss(model,x,y__)

        gradient = tape.gradient(loss2,x)

        signed_grad = tf.sign(gradient)
        x += .001 * signed_grad

        y__ = model(x)
    M,C = stats.mode(classes[2:], axis=0)
    print( M[0], classes[0] )
    if classes[0] != classes[1]:
        num += 1
        perturbed_accuracy(M[0], classes[0])
        defense_accuracy(M[0], classes[1])


print(num)
print("Equals correct class: {:.3%}".format(perturbed_accuracy.result()))
print("Equals adversarial class: {:.3%}".format(defense_accuracy.result()))




