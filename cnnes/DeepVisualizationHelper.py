from matplotlib import pyplot as plt
from keras.models import Sequential
import numpy as np
from keras.models import Model


def get_model_weights_flatten_(deep_model):
    weights = deep_model.get_weights()
    flatten_weights = list()
    for weight in weights:
        flatten_weights.extend(np.reshape(weight, (1, weight.size)).tolist()[0])

    return flatten_weights


def display_activation(activations, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

def visualization(deep_model: Sequential, params=40, mode='hist'):
    if mode == 'hist':
        weights = get_model_weights_flatten_(deep_model)
        plt.hist(weights, params)
        plt.show()
    if mode == 'activations':
        x_train = params['x_train']
        shape = params['shape']
        layer_outputs = [layer.output for layer in deep_model.layers]
        activation_model = Model(inputs=deep_model.input, outputs=layer_outputs)
        activations = activation_model.predict(x_train.reshape((shape)))
        display_activation(activations, 2, 2, 2)
        plt.show()
