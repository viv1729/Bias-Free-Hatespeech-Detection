# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:31:46 2018
@author: vivenkyan
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import Stemmer

import re
import timeit
import string
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


filepath = "labeled_data.csv"

stopWords = set(stopwords.words("english"))
stemmer = Stemmer.Stemmer('english', 100000)    

EPOCHS = 1000
LEARNING_RATE = .01

IP_DIM = 20
OP_DIM = 1


def clean_text(text):
    ## Remove puncuation
    #text = text.translate(string.punctuation)
    
    ## Clean the text 
    #split based on everything except a-z0-9_'.\-
    #tokens = re.findall("[a-z0-9_'.\-]+", text.lower())
    tokens = text.lower().split()
    
    tokens = [stemmer.stemWord(w) for w in tokens if not w in stopWords and len(w) > 2 and len(w)<20]
    text = " ".join(tokens)
    
    return text


def build_data(filepath):
    df = pd.read_csv(filepath)
    data = pd.DataFrame()
    
    data['hate+offensive_count'] = df['offensive_language']
    data['non-hate_count'] = df['neither']
    
    classes = []
    for index, row in data.iterrows():
        temp = 1 if row['hate+offensive_count'] > row['non-hate_count'] else 0
        classes.append(temp)
        
    data['class'] = classes
    
    #label =  {1:hate, 0:non-hate}
    labels = data['class'].map(lambda x : 1 if int(x) == 1 else 0)
    
    #cleaning text
    data['tweet'] = df['tweet'].map(lambda x: clean_text(x))
    
    #data.to_csv("cleaned_tweets5.csv", index=False)
    
    return (data, labels)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def process_sample(sample):
    
    print("\nGiven sample size:", len(sample))
    
    #Keras tokenizer function to tokenize the strings and 
    #‘texts_to_sequences’ to make sequences of words.

    vocabulary_size = 20000

    #Maximum number of words to work with 
    #if set, tokenization will be restricted to the top nb_words most common words in the dataset).
    tokenizer = Tokenizer(num_words= vocabulary_size)

    #fit_on_texts(texts):
    #Arguments: list of texts to train on.
    #tokenizer.fit_on_texts(data['tweet'])
    tokenizer.fit_on_texts(sample)

    #texts_to_sequences(texts)
    #texts: list of texts to turn to sequences.
    #Return: list of sequences (one per text input).
    
    #sequences = tokenizer.texts_to_sequences(data['tweet'])
    sequences = tokenizer.texts_to_sequences(sample)
    sample = pad_sequences(sequences, maxlen = IP_DIM)
    
    print("Processed sample shape:", sample.shape)
    #print("Sample1:", sample[0])
    
    return sample


#MLP Network architecture

NN_ARCHITECTURE = [
    {"input_dim": IP_DIM, "output_dim": 40, "activation": "relu"},
    {"input_dim": 40, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 40, "activation": "relu"},
    {"input_dim": 40, "output_dim": IP_DIM, "activation": "relu"},
    {"input_dim": IP_DIM, "output_dim": OP_DIM, "activation": "sigmoid"},
]

def init_layers(nn_architecture, seed = 99):
    
    np.random.seed(seed)
    
    #dict of weight and bias for each layer
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        #layers are numbered from 1
        layer_idx = idx + 1
        
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # random initiating the values of the W matrix and vector b for each layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    dZ[Z > 0] = 1
    
    return dZ;


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    #input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    #return activated o/p A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    #caching the information needed for a backward step
    memory = {}
    
    #X vector is the activation for layer 0 
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        
        #activated o/p from the previous layer is i/p for this layer
        A_prev = A_curr
        
        activ_function_curr = layer["activation"]
        
        #params_values is the dictionary returned by function init_layers()
        #Key: wi or bi  Value: array of weight/bias for ith layer
        
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        
        #activated o/p for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        #caching
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    #returning activated o/p and cached memory
    return A_curr, memory


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    #number of training examples
    m = A_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Activation function not defined!!!')
    
    #activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    #derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    
    #derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    
    #derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    #ensuring shape of the prediction vector and labels vector are same: ?? 
    Y = Y.reshape(Y_hat.shape)
    
    #gradient descent start
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
      
        layer_idx_curr = layer_idx_prev + 1
    
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values


def get_cost_value(Y_hat, Y):
    #number of training examples
    m = Y_hat.shape[1]
    
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    
    #np.squeeze: Remove single-dimensional entries from the shape of an array.
    #Returns the input array, but with all or a subset of the dimensions of length 1 removed. 
    #This is always array itself or a view into array.
    return np.squeeze(cost)

#converting probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def update(params_values, grads_values, nn_architecture, learning_rate):

    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=True):
    #initializing W and b for each layer
    params_values = init_layers(nn_architecture, 2)
   
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        #feed-forward 
        Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)
        
        #caching
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        #backpropagation
        grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
        
        #updating weights
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % 50 == 0):
            if(verbose):
                print("Iteration: {:05} - cost: {:.8f} - accuracy: {:.10f}".format(i, cost, accuracy))

    return params_values


#build data
data, labels = build_data(filepath)
print("Data read!!!")

#Splitting
X = data['tweet']
Y = labels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#converting panda.series to numpy array
Y_train = Y_train.values
Y_test = Y_test.values

print("Train data len:", len(X_train), "\nTest data len:", len(X_test))

X_train_seq = process_sample(X_train)
X_test_seq = process_sample(X_test)

X_train_seq_norm = np.array([normalize(v) for v in X_train_seq])
X_test_seq_norm = np.array([normalize(v) for v in X_test_seq])

print(type(X_train), type(X_train_seq), type(X_train_seq_norm), "\n")

#for i in range(10):
#    print(X_train.iat[i], "|Y:", Y_train.iat[i])
    
X_train.head(10)


# Training
print("\n\nTraining....")
start_time = timeit.default_timer()
params_values = train(np.transpose(X_train_seq_norm), np.transpose(Y_train.reshape((Y_train.shape[0], 1))), NN_ARCHITECTURE, EPOCHS, LEARNING_RATE)

print("Training time: %.4f sec." % (timeit.default_timer() - start_time))

# Prediction
print("\n\nTesting....")
start_time = timeit.default_timer()

y_test_hat, _ = full_forward_propagation(np.transpose(X_test_seq_norm), params_values, NN_ARCHITECTURE)

print("Testing time: %.4f sec." % (timeit.default_timer() - start_time))

# Accuracy on test set
acc_test = get_accuracy_value(y_test_hat, np.transpose(Y_test.reshape((Y_test.shape[0], 1))))
print("\nMLP Test set accuracy: %.2f" % (acc_test))




