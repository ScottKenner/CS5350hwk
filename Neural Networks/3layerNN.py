import numpy as np



def sigmoid_output(s):
    return 1 / (1 + np.exp(-s))


def calculateLayer1Z(NN,index):
    summate = 0
    if index == 0:
        return 1
    #print("INMDS",index)
    for w in range(0, len(NN.inputs)):
        #print(((w*(NN.layer_width-1)+index) -1))
        summate += NN.X_weights[((w*(NN.layer_width-1)+index) -1)] * NN.inputs[w]
    #print(summate)
    return sigmoid_output(summate)

def calculateLayer2Z(NN,index):
    summate = 0
    if index == 0:
        return 1

    for w in range(0, NN.layer_width):
        #print(NN.Layer1_weights[(index*w) + 1])
        summate += NN.Layer1_weights[((w*(NN.layer_width-1)+index) -1)] * calculateLayer1Z(NN,w)
    return sigmoid_output(summate)


def calculateLayerY(NN):

    summate = 0
    for i in range(0,NN.layer_width):
        #print(summate)
        summate += NN.Layer2_weights[i]*calculateLayer2Z(NN,i)

    return summate
def NNbackprop(rate = 1, epoch= 10, test_a = True):
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
            NN.Layer2_weights = w3
            NN.Layer1_weights = w2
            NN.X_weights = w1
            NN.inputs = inp
            NN.layer_width = 3
            #print(calculateLayerY(NN))

        def backNN(NN):
            y_star = 1
            calculateLayerY(NN)
            total_loss = 0
            deltaL = calculateLayerY(NN) - y_star

            def determine_path_weight(NN, fr, to):
                # print(fr,to)
                re = (fr * (NN.layer_width - 1) + to) - 1
                #print(re)
                return re

            for inp in range(0,len(NN.inputs)):
                for path in range(0,NN.layer_width):
                    z1x = calculateLayer1Z(NN, path)
                    z1der = z1x * (1 - z1x) * NN.inputs[inp] #* NN.X_weights[determine_path_weight(NN,inp,path)]
                    for alt_paths in range(1,NN.layer_width):#z2layers
                        z2x = calculateLayer2Z(NN, alt_paths)
                        z2der = z2x * (1 - z2x)*NN.Layer1_weights[determine_path_weight(NN,path,alt_paths)]
                        wDl = deltaL * NN.Layer2_weights[alt_paths]
                        total_loss+= wDl * z1der*z2der

            print(total_loss)
            return total_loss

        backNN(NN)


            # z1x = calculateLayer1Z(NN,path)
            # z1der = z1x*(1-z1x)*NN.inputs[inp]
            # for alt_paths in range(1,NN.layer_width-1):#z2layers
            #     z2x = calculateLayer2Z(NN,alt_paths)
            #     z2der = z2x*(1-z2x)*NN.Layer1_weights[alt_paths*(NN.layer_width-1)]
            #     wDl =deltaL*NN.Layer2_weights[alt_paths*(NN.layer_width-1)]
            #     total_loss+= wDl * z1der*z2der
            #     print(total_loss)





            # for  i in range(0,3):
            #     for y in range(0,3):
            #         X_node = Node
            #         X_node.weights.append(w1[y * 2])
            #         X_node.weights.append(w1[y * 2] + 1)
            #         Z1_node =  Node
            #         Z1_node.weights.append(w2[y*2])
            #         Z1_node.weights.append(w2[y * 2]+1)
            #         Z2_node = Node
            #         Z2_node.weights.append(w3[y * 2])
            #         NN.inputs.append(X_node)
            #         NN.Layer1_nodes.append(Z1_node)
            #         NN.Layer2_nodes.append(Z2_node)




class NeuralNet:
    y = []
    Layer1_weights = []
    Layer2_weights = []
    inputs = []
    X_weights = []
    layer_width = 0

    Layer1_cache = []
    Layer2_cache = []
    Y_cache = 0



if __name__ == '__main__':
    NNbackprop()