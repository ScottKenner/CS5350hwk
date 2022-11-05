import math
import numpy as np

def basic_predict(w,x,y,r):
    #print( np.sign(np.matmul(np.transpose(w),x)))
    if y.flat[0] != np.sign(np.matmul(np.transpose(w),x)):
        if y.flat[0] == -1: #mult by -1
            x = np.multiply(x, -1)
        ryx = np.multiply(r,x)
        #print(ryx)
        w = np.add(w,ryx)
    return w

def basic_test(w,x,y):
    if y.flat[0] != np.sign(np.matmul(np.transpose(w), x)):
        return 0
    return 1

def basic(rate = 1, epoch= 10):
    with open("bank-note/train.csv", 'r') as train:
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)
        for e in range(epoch):
            for train_data in train_arr:#for xi,yi
                x = np.array(train_data[:len(train_data) - 1])
                y = np.array(train_data[len(train_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)
                #print(x,y)
                weight = basic_predict(weight,x,y, rate)
                #print(weight)
        #debug: does it train ok?
        total =0
        itera = 0
        for train_data in train_arr:  # for xi,yi
            itera+=1
            x = np.array(train_data[:len(train_data) - 1])
            y = np.array(train_data[len(train_data) - 1])
            if y == 0.0:
                y = np.array(-1)
            total+= basic_test(weight,x,y)

        #print(total/itera)
        total =0
        itera = 0
        with open("bank-note/test.csv", 'r') as test:
            test_arr = []
            for line in test:
                xarr = line.strip().split(',')
                for x in range(0, len(xarr)):
                    xarr[x] = float(xarr[x])
                test_arr.append(xarr)

            for test_data in test_arr:  # for xi,yi
                itera+=1
                x = np.array(test_data[:len(test_data) - 1])
                y = np.array(test_data[len(test_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)
                total+= basic_test(weight,x,y)
        #print(total / itera)
        print('Standard Perceptron error is:',1-(total / itera))
        print(weight)


def voted_test(mw,x,y):
    epsil = 0
    for wc in mw:
        w= wc[0]
        c= wc[1]
        epsil+= c*(np.sign(np.transpose(w), x))
    testing  = np.sign(epsil)
    if y.flat[0] != testing.flat[0]:
        return 0
    return 1

def voted(r = 1, epoch= 10):
    with open("bank-note/train.csv", 'r') as train:
        meta_weights = [] #array of weights
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            train_arr.append(xarr)
        weight = [0] * (len(xarr) -1)
        corr_count = 1
        m_index =0
        for e in range(epoch):
            for train_data in train_arr:#for xi,yi
                x = np.array(train_data[:len(train_data) - 1])
                y = np.array(train_data[len(train_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)
                check =  np.matmul(np.transpose(weight), x)

                if y.flat[0] == -1:  # mult by -1
                    check = np.multiply(-1,check)
                if check <= 0:
                    meta_weights.append([weight,corr_count])
                    m_index+=1
                    corr_count = 1
                    ryx = x
                    if y.flat[0] == -1:
                        ryx= np.multiply(x,-1)
                    ryx = np.multiply(r, ryx)

                    weight = np.add(weight,ryx)
                    #print(weight)
                else:
                    corr_count+=1
        #print(meta_weights) #UNCOMMENT ME
        print('Skipping voted weights due to its large size, uncomment line 119 in Perceptron.py for that, or see voted_output.txt for an example output')
        itera =0
        total =0
        with open("bank-note/test.csv", 'r') as test:
            test_arr = []
            for line in test:
                xarr = line.strip().split(',')
                for x in range(0, len(xarr)):
                    xarr[x] = float(xarr[x])
                test_arr.append(xarr)

            for test_data in test_arr:  # for xi,yi
                itera += 1
                x = np.array(test_data[:len(test_data) - 1])
                y = np.array(test_data[len(test_data) - 1])
                if y == 0.0:
                    y = np.array(-1)
                total += voted_test(meta_weights, x, y)
        print('Voted error is:',1-(total / itera))
            #print(weight)

def average_test(a,x,y):
    testing = np.sign(np.matmul(np.transpose(a),x))
    if y.flat[0] != testing:
        return 0
    return 1

def average(r = 1, epoch= 10):
    with open("bank-note/train.csv", 'r') as train:
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)
        a = weight
        for e in range(epoch):
            for train_data in train_arr:#for xi,yi
                x = np.array(train_data[:len(train_data) - 1])
                y = np.array(train_data[len(train_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)


                check = np.matmul(np.transpose(weight), x)
                if y.flat[0] == -1:  # mult by -1
                    check = np.multiply(-1, check)
                if check <= 0:
                    ryx = x
                    if y.flat[0] == -1:
                        ryx = np.multiply(x, -1)
                    ryx = np.multiply(r, ryx)

                    weight = np.add(weight, ryx)
                a = np.add(a, weight)

        #print(total/itera)
        total =0
        itera = 0
        with open("bank-note/test.csv", 'r') as test:
            test_arr = []
            for line in test:
                xarr = line.strip().split(',')
                for x in range(0, len(xarr)):
                    xarr[x] = float(xarr[x])
                test_arr.append(xarr)

            for test_data in test_arr:  # for xi,yi
                itera+=1
                x = np.array(test_data[:len(test_data) - 1])
                y = np.array(test_data[len(test_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)
                total+= average_test(a,x,y)
        print('Average error is:',1-(total / itera))
        #print(weight)
        print(a)


if __name__ == '__main__':
    basic()
    voted()
    average()