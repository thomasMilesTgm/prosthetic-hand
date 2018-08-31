import numpy as np
import theano
import theano.tensor as T
from theano.tensor import fft
import lasagne
import os, glob

# Number of input data streams
NUM_INPUTS = 1

# FFT sapling rate

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128

# Number of gestures to learn
NUM_LABELS = 1

# Raw readings from the EMG
raw_data = []

def load_data(data_dir="/home/tmiles/data/prosthetic/dummy/"):
    """
    Loads raw data
    :param data_dir:
    :return:
    """
    for filename in glob.glob(os.path.join(data_dir, '*')):
        with np.load(filename) as f:
            raw_data.append(f)


def gen_training_data(raw_data):
    """
    Performs fast fourier transforms on the raw data, in batch size arrays
    :param raw_data: raw numpy data
    :return: Fourier transformed data in
    """


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, None, NUM_INPUTS))

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)

    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the softmax nonlinearity to
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, NUM_LABELS)
    l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=NUM_LABELS, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    # In order to produce the probability distribution of the prediction, we compile a function called probs.
    probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)




    print("Training ...")

    p = 0
    try:
        for it in range(int(data_size * num_epochs / BATCH_SIZE)):

            avg_cost = 0
            avg_cost += train(x, y)

            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
