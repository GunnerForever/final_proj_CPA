from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def train_model(model, captions, img_feats, pad_idx, args, valid):
    '''Trains model and returns model statistics'''
    stats = []
    try:
        for epoch in range(args.epochs):
            stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e
        
    return stats


def test_model(model, captions, img_feats, pad_idx, args):
    '''Tests model and returns model statistics'''
    perplexity, accuracy = model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)
    return perplexity, accuracy

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_results():
    pass

def main(epochs):
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.
    
    Students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    :return: None
    '''

    LOCAL_TRAIN_FILE = 'data/train'
    LOCAL_TEST_FILE = 'data/test'

    CLASSES = [3, 5]
    # TODO: assignment.main() pt 1
    # Load your testing and training data using the get_data function

    data = get_data(LOCAL_TRAIN_FILE, CLASSES)
    test_data = get_data(LOCAL_TEST_FILE, CLASSES)

    # Initialize your model and optimizer
    #model = 
    optimizer = tf.optimizers.Adam(0.003)
    # Train your model
    for i in range(epochs):
        train(model, optimizer, train_imps, train_labradors)
    # Test your model
    acc = test(model, test_imps, test_labradors)
    print("Final Acc: " + str(acc))
    visualize_loss(model.loss_list)
    return 


if __name__ == '__main__':
    main(5)
