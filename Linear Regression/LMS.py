
import matplotlib.pyplot as plt
import numpy as np


def compute_gradient(xarr,x, weights):
    epsil = 0
    y = float(xarr[len(xarr)-1])
    #for x in range(0,len(xarr)-1):
    dot_prod = 0
    for ite in range(0, len(xarr) - 1):
        dot_prod += weights[ite]*float(xarr[ite])
    epsil += (y-(dot_prod))*float(xarr[x])
    return epsil

def update_weight(weight, deltaj, gamma):
    ret_arr = []
    for w in range(0,len(weight)):
        ret_arr.append(weight[w] - deltaj*gamma)

    return ret_arr

def calc_error(tarr, weight):
    error = 0
    for m in tarr:
        for i in range(0, len(m)-1):
            error+= .5*((float(m[len(m)-1]) - float(m[i])*float(weight[i]))**2)
    return error


def batch_descent(t, gamma):
    with open("../Storage/concrete/train.csv", 'r') as train:
        train_arr = []
        for line in train:
            xarr = line.strip().split(',')
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)
            deltajarr = [0] * (len(xarr) -1)
            error = 1

        for iter in range(0,t):
            tempDjarr = deltajarr
            for xarr in train_arr:#for m
                for x in range(0, len(xarr) - 1):#calculate each xij's delta
                    tempDjarr[x] -= ((1/len(train_arr)) * compute_gradient(xarr,x, weight))

            old_weight = calc_error(train_arr, weight)

            tempW = [0] * (len(xarr) -1)
            for dj in range(0, len(xarr) - 1):
                tempW[dj] = weight[dj] - (gamma*tempDjarr[dj])
            # print(weight)
            # print(gamma)
            # print(abs(calc_error(train_arr, tempW)- old_weight))
            # print(old_weight)
            if abs(calc_error(train_arr, tempW) - old_weight) < .000001:
                print("DONE!")
                # plt.ylabel('Error diffrence b/w old w and new w error')
                # plt.xlabel('Gamma')
                # plt.show() UNCOMMENT FOR PLOT
                print("Showing batch gamma, delta, and weight")
                print(gamma)
                print(abs(calc_error(train_arr, tempW) - old_weight))
                print(weight)
                break

            if (calc_error(train_arr, tempW) - old_weight) > 0:
                gamma = gamma/2
                deltajarr = [0] * (len(xarr) - 1)
                plt.scatter(gamma, abs(calc_error(train_arr, tempW) - old_weight))

                continue
            else:
                deltajarr = [0] * (len(xarr) - 1)
                weight = tempW



            #plt.plot(weight[0], iter)

    #plt.show()
                #
                #
                # deltajarr.append( compute_gradient(xarr, weight))
                # weight_next = update_weight(weight,deltaj, gamma)
                # print(weight_next)
                # weight = weight_next


def sto_compute_gradient(xarr,x, weights):
    epsil = 0
    y = float(xarr[len(xarr)-1])
    #for x in range(0,len(xarr)-1):
    dot_prod = 0
    for ite in range(0, len(xarr) - 1):
        dot_prod += weights[ite]*float(xarr[ite])

    return (y-(dot_prod))*float(xarr[x])


def sto_calc_error(tarr, weight):
    error = 0
    for m in tarr:
        for i in range(0, len(m)-1):
            error+= .5*((float(m[len(m)-1]) - float(m[i])*float(weight[i]))**2)
    return error


def sto_descent(t, gamma):
    with open("../Storage/concrete/train.csv", 'r') as train:
        train_arr = []
        for line in train:
            xarr = line.strip().split(',')
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)
            deltajarr = [0] * (len(xarr))
            error = 1
        import random
        for iter in range(0,t):

            tempDjarr = weight
            xarr = random.choice(train_arr)
            old_weight = calc_error(train_arr, weight)
            old_calc = calc_error(train_arr, weight) - old_weight
            for x in range(0, len(xarr) - 1): #calculate each xij's delta
                tempDjarr[x] = (tempDjarr[x] + gamma * (sto_compute_gradient(xarr,x, weight)))


            weight = tempDjarr



            tempW = weight


            #print(weight)
            # print(gamma)
            # print(abs(calc_error(train_arr, tempW)- old_weight))
            # print(old_weight)
            if abs(calc_error(train_arr, tempW) - old_weight) < .000001:
                print("DONE!")
                # plt.ylabel('Error diffrence b/w old w and new w error')
                # plt.xlabel('Gamma')
                # plt.show() UNCOMMENT FOR PLOT
                print("Showing stochastic gamma, delta, and weight")
                print(gamma)
                print(abs(calc_error(train_arr, tempW) - old_weight))
                print(weight)
                break

            if (abs(calc_error(train_arr, tempW) - old_weight)) > old_calc:
                gamma = gamma/1.1
                deltajarr = [0] * (len(xarr) - 1)
                plt.scatter(gamma, abs(calc_error(train_arr, tempW) - old_weight))

                continue
            else:
                deltajarr = [0] * (len(xarr) - 1)
                weight = tempW

def opt_calc():
    with open("../Storage/concrete/train.csv", 'r') as train:
        train_arr = []
        ytrain = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            train_arr.append(xarr[0:len(xarr)-1])
            ytrain.append(xarr[len(xarr)-1])
        nxarr = np.array(train_arr)
        nyarr = np.array(ytrain)
        inv  = np.linalg.inv(np.matmul(np.transpose(nxarr),nxarr))
        xmult = np.matmul(inv,np.transpose(nxarr))
        #print(np.shape(xmult))
        #print(np.shape(nyarr))
        print("Showing Optimal  weight output")
        print(np.matmul(xmult,nyarr))
def opt_calc_hwk():
    with open("../Storage/concrete/hwk.csv", 'r') as train:
        train_arr = []
        ytrain = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                xarr[x]= float(xarr[x])
                print(xarr)




            train_arr.append(xarr[0:len(xarr)-1])
            ytrain.append(xarr[len(xarr)-1])
        nxarr = np.array(train_arr)
        nyarr = np.array(ytrain)
        inv  = np.linalg.inv(np.matmul(np.transpose(nxarr),nxarr))
        xmult = np.matmul(inv,np.transpose(nxarr))
        print("Showing Optimal  weight output")
        print(np.matmul(xmult,nyarr))


def sto_descent_hwk(t, gamma):
    with open("../Storage/concrete/hwk.csv", 'r') as train:
        train_arr = []
        for line in train:
            xarr = line.strip().split(',')
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)
            deltajarr = [0] * (len(xarr) -1)
            error = 1
        import random
        for iter in range(0,5):

            tempDjarr = weight
            xarr = train_arr[iter]
            old_weight = calc_error(train_arr, weight)
            old_calc = calc_error(train_arr, weight) - old_weight
            for x in range(0, len(xarr) - 1): #calculate each xij's delta
                tempDjarr[x] = (tempDjarr[x] + (gamma * (sto_compute_gradient(xarr,x, weight))))

            print(tempDjarr)
            weight = tempDjarr



if __name__ == '__main__':
    batch_descent(1000000, .5)
    sto_descent(10000, .5)
    opt_calc()
    #sto_descent_hwk(5,.1)