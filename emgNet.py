import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano.tensor import fft
import lasagne
import os, glob

# Data sampling rate (ms)
SAMPLE_RATE = 100

# FFT rate (ms)
FFT_RATE = 600

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 10

# Block Size (number of points in a block)
BLOCK_SIZE = 180

# Number of blocks in a batch
BATCH_SIZE = 180

# Number of gestures to learn
NUM_LABELS = 11

# Raw readings from the EMG
raw_data = []

# Number of input channels including timestamps
NUM_CHANNELS = 8

# length of an epoch (sec)
EPOCH_TIME = 60


def load_data(data_dir="/home/tmiles/data/prosthetic/dummy/"):
    """
    Loads raw data
    :param data_dir:
    :return:
    """
    for filename in glob.glob(os.path.join(data_dir, '*')):
        with np.load(filename) as f:
            raw_data.append(f)


def loadTrainingData(data, labels, sample_rate, block_size, epoch_length, return_label=True):
    """
    Loads data into discrete blocks with unique labels.

    :param data: file path to datafile
    :param labels: file path to the label file
    :param block_size: number of points in a single data block
    :param epoch_length: total time for an epoch (ms)
    :param sample_rate: time interval (ms) to sample data
    :param return_label: should this data be labeled? default == True

    :return x: input data (num_blocks, num_channels, block_size)

    :return y: labels (num_blocks, num_labels)
    """
    num_blocks = int(epoch_length*1000 / (sample_rate*block_size))   # number of data blocks
    label_dict = {}

    x = np.zeros((num_blocks, NUM_CHANNELS, block_size))
    y = np.zeros((num_blocks, NUM_LABELS))
    label = np.zeros(shape=NUM_LABELS)

    # open data and label files
    d_raw = np.load(data)
    l_raw = np.load(labels)

    t0 = d_raw[0][0]    # initial timestamp
    t_last = t0         # most recent timestamp

    j = 0               # raw data index
    i = 0               # block index
    int_ix = 0


    for time in d_raw[0]:
        # print((time - t0)/1000)

        # epoch is full
        if (time - t0)/1000 >= epoch_length or i >= num_blocks:
            return x, y

        for n in range(block_size):
            # only add the point if enough time has passed since last tick
            if d_raw[0, j] - t_last < sample_rate:
                j += 1
                pass

            # if we have filled a block, start a new one
            if int_ix >= block_size:
                if return_label:
                    label = label / block_size
                    # for l in range(len(label)):     # average label
                    #     label[l] = l / j
                    y[i] = label
                    label = np.zeros(shape=NUM_LABELS)

                # increment block index
                int_ix = 0
                i += 1
                break

            # make sure we havent run out of data
            if j >= len(d_raw[0]):

                if return_label:
                    label = label / int_ix
                    # for l in range(len(label)):     # average label
                    #     label[l] = l / j

                    y[i] = label

                return x, y

            # Otherwise add data and labels
            for k in range(NUM_CHANNELS):

                x[i, k, int_ix] = d_raw[k, j]

            if return_label:
                # Since label may change during a block, we're going to average the label, keep a cumulative sum
                this_label, label_dict = l_to_out(l_raw[1][j], label_dict)
                label += this_label

            # increment raw data index
            int_ix += 1
            j += 1
            t_last = time
    return x, y


def l_to_out(label, label_dict):
    """
    Converts a label (unicode value) to an array of output values expected of the network
    :param label: label to find index for (string)
    :param label_dict: dictionary of existing labels
    :return out: output layer array
    """
    keys = label_dict.keys()
    out = np.zeros(NUM_LABELS)
    assigned = False

    if len(keys) == 0:
        label_dict = {label: 1}
        out[0] = 1
        return out, label_dict

    else:
        for i in range(len(keys)):

            if list(keys)[i] == label:
                out[i] = 1
                assigned = True

        # if not all possible labels have been assigned, add this to the dict
        if len(keys) < NUM_LABELS and not assigned:

            new_label = {label: 1}
            label_dict.update(new_label)
            out[len(keys)-1] = 1

        return out, label_dict


def out_to_l(out, dict):
    """
    Returns the label name which has the highest activation value from the output layer
    :param out: network output
    :param dict: key dictionary
    :return:
    """
    keys = dict.keys()
    activation = 0
    active = None

    for i in range(len(keys)):
        if out[i] > activation:
            activation = out[i]
            active = i

    if active:
        return chr(out[active])


def test_load(sample_rate, block_size, epoch_length):
    data = "/home/tmiles/data/prosthetic/dummy/1536640459.6213417/data.npy"
    label = "/home/tmiles/data/prosthetic/dummy/1536640459.6213417/labels.npy"

    x, y = loadTrainingData(data, label, sample_rate, block_size, epoch_length)
    print("Data Loaded")
    print("x shape (num_blocks, num_channels, block_size): (" + str(len(x)) + "," + str(len(x[0])) + "," + str(len(x[0,0])) + ")")
    print("y shape (num_blocks, num_labels): (" + str(len(y)) + "," + str(len(y[0])) + ")")
    print("x, y = \n")
    for i in range(len(x)):
        if x[i,0,0] == 0:
            break
        print(len(x))
        print("____________________________________________________")
        print(y[i])

        print("____________________________________________________")
        print("____________________________________________________")

def iterate_minibatches(x_train, y_train, batch_size=BATCH_SIZE):

    x = []
    y = []
    i=0

    while i < len(x_train[0,0]):
        x.append(np.array(x_train[i:(i+batch_size):1, :, :]))
        y.append(np.array(y_train[i:(i+batch_size):1, :]))
        i += batch_size

    return x, y

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, NUM_CHANNELS, BLOCK_SIZE))

    batchsize, seqlen, _ = l_in.input_var.shape

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, num_units=N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, num_units=N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)

    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the softmax nonlinearity to
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, NUM_LABELS)
    l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=NUM_LABELS, W=lasagne.init.Normal(),
                                      nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.lvector('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    # In order to produce the probability distribution of the prediction, we compile a function called probs.
    probs = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

    print("Training ...")
    data = "/home/tmiles/data/prosthetic/dummy/1536640459.6213417/data.npy"
    label = "/home/tmiles/data/prosthetic/dummy/1536640459.6213417/labels.npy"

    x, y = loadTrainingData(data, label, SAMPLE_RATE, BLOCK_SIZE, EPOCH_TIME)

    p = 0
    try:
        for epoch in range(NUM_EPOCHS):

            avg_cost = 0
            for batch in iterate_minibatches(x, y):

                intputs, targets = batch
                avg_cost += train(intputs, targets)

            print("Epoch {} average loss = {}".format(1.0 * PRINT_FREQ / NUM_EPOCHS * BLOCK_SIZE, avg_cost / PRINT_FREQ))

    except KeyboardInterrupt:
        pass

def test_iteratemini():
    data = "/home/tmiles/data/prosthetic/dummy/1536640459.6213417/data.npy"
    label = "/home/tmiles/data/prosthetic/dummy/1536640459.6213417/labels.npy"

    x, y = loadTrainingData(data, label, SAMPLE_RATE, BLOCK_SIZE, EPOCH_TIME)
    x_mini, y_mini = iterate_minibatches(x, y)
    for batch in x_mini:
        print(batch)

    print(str(len(x_mini)) + " batches made")


if __name__ == '__main__':
    main()
    # test_load(sample_rate=100, block_size=10, epoch_length=60)
    # test_iteratemini()



