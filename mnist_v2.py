import tensorflow as tf

BATCH_SIZE = 32
TEST_BATCH_SIZE = 3200

ENSEMBLE_SIZE = 10
PRUNE_FACTOR = 0.1

# Enable eager so that dataset iterator will work
# tf.enable_eager_execution()
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


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


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

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)

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
        y_ = tf.argmax(input=model_copy(x), axis=1)
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
for epoch in range(num_epochs):
    train_dataset = train_dataset.shuffle(100000)
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()


    # Training loop - using batches of 32
    for x, y in train_dataset:
        i += BATCH_SIZE

        if LAYER_TUPLES:
            loss_value = train_step(
                                x,
                                y,
                                model=model,
                                model_copy=model_copy,
                                ensemble_size=ENSEMBLE_SIZE,
                                prune_factor=PRUNE_FACTOR,
                                layer_tuples=LAYER_TUPLES,
                                num_layers=NUM_LAYERS
                            )
        else:
            loss_value, LAYER_TUPLES, NUM_LAYERS = train_step(
                                                            x,
                                                            y,
                                                            model=model,
                                                            model_copy=model_copy,
                                                            ensemble_size=ENSEMBLE_SIZE,
                                                            prune_factor=PRUNE_FACTOR,
                                                            layer_tuples=None,
                                                            num_layers=None
                                                        )
        epoch_loss_avg(loss_value)


        y_ = make_predictions(x, y, model=model, model_copy=model_copy, ensemble_size=ENSEMBLE_SIZE, prune_factor=PRUNE_FACTOR, layer_tuples=LAYER_TUPLES, num_layers=NUM_LAYERS)
        epoch_accuracy(y_, y)

        if i % 128 == 0:
            print("Epoch {:03d}, Examples processed {:03d}, Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    i,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 1 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))



test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
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
    test_accuracy(y_, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

model.save('mnist.h5')


