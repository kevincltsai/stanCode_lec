"""
File: titanic_deep_nn.py
Name: 
-----------------------------
This file demonstrates how to create a deep
neural network (5 layers NN) to train our
titanic data. Your code should use all the
constants and global variables.
You should see the following Acc if you
correctly implement the deep neural network
Acc: 0.8431372549019608
-----------------------------
X.shape = (N0, m)
Y.shape = (1, m)
W1.shape -> (N0, N1)
W2.shape -> (N1, N2)
W3.shape -> (N2, N3)
W4.shape -> (N3, N4)
W5.shape -> (N4, N5)
B1.shape -> (N1, 1)
B2.shape -> (N2, 1)
B3.shape -> (N3, 1)
B4.shape -> (N4, 1)
B5.shape -> (N5, 1)
"""

from collections import defaultdict
import numpy as np

# Constants
TRAIN = 'titanic_data/train.csv'     # This is the filename of interest
NUM_EPOCHS = 40000                   # This constant controls the total number of epochs

ALPHA = 0.01                         # This constant controls the learning rate α
L = 5                                # This number controls the number of layers in NN
NODES = {                            # This Dict[str: int] controls the number of nodes in each layer
    'N0': 6,
    'N1': 5,
    'N2': 4,
    'N3': 3,
    'N4': 2,
    'N5': 1
}


def main():
    """
    Print out the final accuracy of your deep neural network!
    You should see 0.8431372549019608
    """
    X_train, Y = data_preprocessing()
    _, m = X_train.shape
    print('Y.shape', Y.shape)
    print('X.shape', X_train.shape)
    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = normalize(X_train)
    ####################################
    #                                  #
    #              TODO:               #
    #                                  #
    ####################################

    weights, biases = neural_network(X, Y)
    
    # The last forward prop
	
    #A0 = X
    A = X
    for i in range(1, L+1):
        A = weights['W' + str(i)].T.dot(A) + biases['B'+str(i)]
        A = np.maximum(0,A)
	
    H = 1 / (1+np.exp(-A))
    predictions = np.where(H>0.5,1,0)
    acc = np.equal(predictions, Y)
    num_acc = np.sum(acc)
    print('Training Acc:', num_acc/m)

def normalize(X):
    """
    :param X: numpy_array, the dimension is (num_phi, m)
    :return: numpy_array, the values are normalized, where the dimension is still (num_phi, m)
    """
    min_array = np.min(X, axis=1, keepdims=True)
    max_array = np.max(X, axis=1, keepdims=True)
    return (X - min_array) / (max_array - min_array)


def neural_network(X, Y):
    """
    :param X: numpy_array, the array holding all the training data
    :param Y: numpy_array, the array holding all the ture labels in X
    :return (weights, bias): the tuple of parameters of this deep NN
             weights: Dict[str, float], key is 'W1', 'W2', ...
                                        value is the corresponding float
             bias: Dict[str, float], key is 'B1', 'B2', ...
                                     value is the corresponding float
    """
    np.random.seed(1)
 
    # Initialize all the weights and biases
    #####################################
    #                                   #
    #               TODO:               #
    #                                   #
    #####################################
    


    print_every = 1000
    _,m = X.shape
    weights = {}
    biases = {}
    for i in range(1, L + 1): #{1,2,3,4,5,6}
        weights['W'+str(i)] = np.random.rand(NODES['N' + str(i-1)], NODES['N'+str(i)]) - 0.5
        biases['B'+str(i)] = np.random.rand(NODES['N'+str(i)],1) - 0.5


    k_lst = {}
    a_lst = {'A0':X}

    # W > K > A
    #NUM_EPOCHS = 10000
    for epoch in range(NUM_EPOCHS):
        # Forward Pass
        # TODO:
        for i in range(1, L+1): #{1,2,3,4,5}
            #dim check @ i  = 1
            # W1(6,5).T, A0 = (6,714) => (5,714) B1 = (5,1)
            k_lst['K'+str(i)] = np.dot(weights['W'+ str(i)].T,a_lst['A'+str(i-1)]) + biases['B' + str(i)]
            if i <= 4:
                a_lst['A'+str(i)] = np.maximum(0,k_lst['K' + str(i)])
        
        #i stops at 5        
        #k_lst['K'+str(i)] = np.dot(weights['W'+ str(i)].T,a_lst['A'+str(i-1)]) + biases['B' + str(i)]
        scores = k_lst['K'+str(i)]
        H = 1 / (1 + np.exp(-scores))
                
        #print(H)
        J = 1/m * np.sum(-(Y * np.log(H) + (1-Y) * np.log(1-H)))
        #print(J)  
        if epoch % print_every == 0:
            #continue
            print('epoch :', epoch , ' Cost :', J)
        # Backward Pass
        # TODO:

        dK_lst = {}
        dW_lst = {}
        dA_lst = {}
        dB_lst = {}

        #i stops at 5        
        dK_lst['dK5'] = 1/m * np.sum(H-Y, axis=0, keepdims = True)
        dW_lst['dW5'] = np.dot(a_lst['A'+str(i-1)], dK_lst['dK'+str(i)].T)
        dB_lst['dB5'] = np.sum(dK_lst['dK'+str(i)], axis = 1, keepdims=True)

        for i in reversed(range(1,5)):
            #i starts from 4 to 1
            
            dA_lst['dA' + str(i)] = np.dot(weights['W'+str(i+1)], dK_lst['dK'+str(i+1)])
            dK_lst['dK' + str(i)] = dA_lst['dA'+str(i)] * np.where(k_lst['K'+str(i)]>0,1,0)
            dW_lst['dW' + str(i)] = np.dot(a_lst['A'+str(i-1)], dK_lst['dK' + str(i)].T)
            dB_lst['dB' + str(i)] = np.sum(dK_lst['dK' + str(i)], axis=1, keepdims= True)

        # Updates all the weights and biases
        # TODO:
        
        for i in range(1, L + 1):
            weights['W'+ str(i)] -= ALPHA * dW_lst['dW' + str(i)]
            biases['B' + str(i)] -= ALPHA * dB_lst['dB' + str(i)]
            #k_lst['K' + str(i)] -= ALPHA * dK_lst['dK' + str(i)]
            #if i <= 4 :
            #    a_lst['A'+str(i)] -= ALPHA * dA_lst['dA' + str(i)]
        
    return weights, biases


def data_preprocessing(mode='train'):
    """
    :param mode: str, indicating if it's training mode or testing mode
    :return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
    """
    data_lst = []

    label_lst = []
    first_data = True
    if mode == 'train':
        with open(TRAIN, 'r') as f:
            for line in f:
                data = line.split(',')
                # ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
                if first_data:
                    first_data = False
                    continue
                if not data[6]:
                    continue
                label = [int(data[1])]
                if data[5] == 'male':
                    sex = 1
                else:
                    sex = 0
                # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
                passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
                data_lst.append(passenger_lst)
                label_lst.append(label)
    else:
        pass
    return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
    main()
