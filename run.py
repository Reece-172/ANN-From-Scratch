import numpy as np
import sys
import sklearn.decomposition as skd
import sklearn.preprocessing as skp

def normalize(data):
    return skp.normalize(data)

def pca(data, n): #perform PCA on data to reduce dimensionality (change the n_components / features to see how it affects the results)
    data = normalize(data)
    pca = skd.PCA(n_components=n)
    pca.fit(data)
    return pca.transform(data)

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z), axis=0)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def get_predictions(A3): 
    return np.argmax(A3,0)

def forward_prop(X, w1, b1, w2, b2, w3, b3): #forward propagation
    Z1 = np.dot(w1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(w3, A2) + b3
    A3 = softmax(Z3)
    return A3

w1 = np.loadtxt('w1.txt')
b1 = np.loadtxt('b1.txt')
w2 = np.loadtxt('w2.txt')
b2 = np.loadtxt('b2.txt')
w3 = np.loadtxt('w3.txt')
b3 = np.loadtxt('b3.txt')

n = 50
data = np.loadtxt(sys.stdin)
data = pca(data, n)
testing_data = data[1900:].T #get sample from the end of the data set

# output = get_predictions(forward_prop(testing_data, w1, b1, w2, b2, w3, b3))
mult = np.dot(w1, testing_data)+b1 # this gives an error but it works in the notebook
sys.stdout.write(str(mult.shape))