import numpy as np
import sys

def normalize(data):#normalize the data
    return data - data.mean() / data.std()

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z), axis=0)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def get_predictions(A3): 
    return np.argmax(A3,0)

def get_accuracy(predictions, Y):
    accuracy = np.sum(predictions == Y) / Y.size
    return accuracy * 100

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
b1 = b1.reshape(20,1)
b2 = b2.reshape(20,1)
b3 = b3.reshape(10,1)
labels = np.loadtxt('labels.txt')
labels = labels.reshape(2000,1)
eigens = np.loadtxt("eigens.txt")
allData = np.loadtxt(sys.stdin)
allData = normalize(allData)
predictions = []
for data in allData:
    scaled_data = np.dot(data, eigens)
    test_data = scaled_data.reshape(50,1)
    output = get_predictions(forward_prop(test_data, w1, b1, w2, b2, w3, b3))
    predictions.append(output)
    # sys.stdout.write(str(output))

predictions = np.array(predictions)
print(get_accuracy(predictions, labels))
#python3 run.py<test.txt
