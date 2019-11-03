import tensorflow as tf

BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

ENSEMBLE_SIZE = 10
PRUNE_FACTOR = 0.1

# Enable eager so that dataset iterator will work
tf.enable_eager_execution()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
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


model = tf.keras.models.load_model('mnist.h5')



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

def prune_weights(weights, factor=0.1, tuples=None, num_layers=None):
    if tuples:
        for j in range(num_layers):
            for it in tuples[j]:
                if random.random() < factor:
                    weights[j][it] = 0
        return None, None
    else:
        tuples = []
        for j in range(len(weights)):
            s = weights[j].shape
            t = []
            out = [[]]
            for k in s:
                out2 = out.copy()
                out.clear()
                for v in out2:
                    out.extend([v + [q] for q in range(k)] )
            out = [tuple(v) for v in out]
            tuples.append(out)
            for it in out:
                if random.random() < factor:
                    weights[j][it] = 0
        return tuples, len(tuples)




def train_step(x, y, model=None, model_copy=None, ensemble_size=None, prune_factor=0.1, layer_tuples=None, num_layers=None):

    if not num_layers:
        return_type = 1
    else:
        return_type = 0

    loss_values = []
    for n in range(0,ensemble_size):
        weights = model.get_weights()

        if layer_tuples:
            prune_weights(weights, factor=prune_factor, tuples=layer_tuples, num_layers=num_layers)
        else:
            layer_tuples, num_layers = prune_weights(weights, factor=0.1, tuples=None, num_layers=None)

        model_copy.set_weights(weights)
        loss_value, grads = grad(model_copy, x, y)
        grads = [g/ensemble_size for g in grads]
        loss_values.append(loss_value)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)
    
    if return_type == 1:
        return np.average(loss_values), layer_tuples, num_layers
    else:
        return np.average(loss_values)

def make_predictions(x, y, model=None, model_copy=None, ensemble_size=None, prune_factor=0.1, layer_tuples=None, num_layers=None):

    if not num_layers:
        return_type = 1
    else:
        return_type = 0

    predictions = []
    for n in range(0,ensemble_size):
        weights = model.get_weights()

        if layer_tuples:
            prune_weights(weights, factor=prune_factor, tuples=layer_tuples, num_layers=num_layers)
        else:
            layer_tuples, num_layers = prune_weights(weights, factor=0.1, tuples=None, num_layers=None)

        model_copy.set_weights(weights)
        y_ = tf.argmax(model_copy(x), axis=1)
        y_ = tf.dtypes.cast(y_,tf.uint8)
        predictions.append(y_)

    predictions = np.vstack(predictions)

    #return mode and count of predictions
    M,C = stats.mode(predictions, axis=0)

    if return_type == 1:
        return M[0], layer_tuples, num_layers
    else:
        return M[0]


import numpy as np
from scipy import stats
model_copy= tf.keras.models.clone_model(model)
import itertools
import random




perturbed_accuracy = tfe.metrics.Accuracy()
defense_accuracy = tfe.metrics.Accuracy()


i = 0
for (x, y) in test_dataset:
    i += 1
    if i == 10:
        break;
    y__ = model(x)

    with tf.GradientTape() as tape:
        tape.watch(x)
        loss2 = scc(y,y__)

    #print(x)
    #exit()

    gradient = tape.gradient(loss2,x)
    print ("gradient",gradient)
    exit()

    signed_grad = tf.sign(gradient)
    x += .001 * signed_grad

    y_ = model(x)
    perturbed_accuracy(y_, y__)

    y_ = make_predictions(
            x,
            y,
            model=model,
            model_copy=model_copy,
            ensemble_size=ENSEMBLE_SIZE,
            prune_factor=PRUNE_FACTOR,
            layer_tuples=LAYER_TUPLES,
            num_layers=NUM_LAYERS
        )
    defense_accuracy(y_, y__)

print("Perturbed error rate: {:.3%}".format(1 - perturbed_accuracy.result()))
print("Defense error rate: {:.3%}".format(1 - defense_accuracy.result()))




