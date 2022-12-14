import numpy as np

def sigmoid_output(s):
    1 / (1 + np.exp(-s))
def calculateLayer1Z(NN, Z1_node, index):
    summate = 0
    if index == 0:
        return 1

    for w in range(0, len(NN.inouts)):
        summate += NN.Layer1_nodes[w].weights[index] * calculateLayer1Z(w)
    return sigmoid_output(summate)

def calculateLayer2Z(NN, Z2_node, index):
    summate = 0
    if index == 0:
        return 1

    for w in range(0, len(NN.Layer1_nodes)):
        summate += NN.Layer1_nodes[w].weights[index] * calculateLayer1Z(w)
    return sigmoid_output(summate)
def calculateLayerY(NN, Z2_node, index):
    summate = 0
    for w in range(0,len(NN.Layer1_nodes)):
        summate += NN.Layer2_nodes[w].weights[index]*calculateLayer2Z(w)
    return summate
def NNbackprop(rate = 1, epoch= 10, test_a = False):
    with open("bank-note 2/train.csv", 'r') as train:
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)


        if test_a:#we are doing problem A and not the training array
            NN = NeuralNet
            inp = [1,1,1]
            w1 = [-1,1,-2,2,-3,3]
            w2 = [-1,1,-2,2,-3,3]
            w3 = [-1,2,-1.5]





            for  i in range(0,3):
                for y in range(0,3):
                    X_node = Node
                    X_node.weights.append(w1[y * 2])
                    X_node.weights.append(w1[y * 2] + 1)
                    Z1_node =  Node
                    Z1_node.weights.append(w2[y*2])
                    Z1_node.weights.append(w2[y * 2]+1)
                    Z2_node = Node
                    Z2_node.weights.append(w3[y * 2])
                    NN.inputs.append(X_node)
                    NN.Layer1_nodes.append(Z1_node)
                    NN.Layer2_nodes.append(Z2_node)




class NeuralNet:
    y = []
    Layer1_nodes = []
    Layer2_nodes = []
    inputs = []
class Node:
    weights = []
    def sigmoid_output(self):


if __name__ == '__main__':
    NNbackprop()