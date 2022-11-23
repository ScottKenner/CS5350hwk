import math
import numpy as np
import random
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
C = [100/873,500/873,700/873]
def update_gamma_basic(gamma0,alpha,t):
    return gamma0/(1+(gamma0/alpha)*t)
def update_gamma_2(gamma0,alpha,t):
    return gamma0/(1+t)
def basic(rate = 1, alpha=1,epoch= 100,c_index = 0, altg=False):
    with open("bank-note/train.csv", 'r') as train:
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            train_arr.append(xarr)
            weight = [0] * (len(xarr) -1)
        weight0 = weight
        for e in range(epoch): #for each epochg
            init_weight = weight
            random.shuffle(train_arr)#shuffle
            for train_data in train_arr:#for xi,yi
                x = np.array(train_data[:len(train_data) - 1])
                y = np.array(train_data[len(train_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)
                if altg:
                    gammat = update_gamma_2(rate, alpha, e)
                else:
                    gammat = update_gamma_basic(rate, alpha, e)
                    #input(gammat)
                if y*np.matmul(np.transpose(weight),x) <= 1:

                    subtr = np.subtract(weight,np.multiply(gammat,weight0))
                   # print(y*x)
                    weight = np.add(subtr,(gammat*C[c_index]*len(train_data)*(y*x)))
                else:
                    weight0 = np.multiply((1.0-gammat),weight0)
            #print(weight, "\n", init_weight)
            if weight.any == np.array(init_weight).any:
                print("convergence at ", e)
                break
            #input()
        #debug: does it train ok?
        #print(weight, "\n-----")
        #input('aaaa')
        total =0
        itera = 0
        for train_data in train_arr:  # for xi,yi
            itera+=1
            x = np.array(train_data[:len(train_data) - 1])
            y = np.array(train_data[len(train_data) - 1])
            if y == 0.0:
                y = np.array(-1)
            total+= basic_test(weight,x,y)
        #print('Subgrad. SVM Train error is:',1-(total / itera), "\nAt C of", C[c_index],"\\\\")
        testret = 1-(total / itera)
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
            avgbias = 0
            for test_data in test_arr:  # for xi,yi
                itera+=1
                x = np.array(test_data[:len(test_data) - 1])
                y = np.array(test_data[len(test_data) - 1])
                if y  == 0.0:
                    y = np.array(-1)
                total+= basic_test(weight,x,y)
                #avgbias+= ((y-np.matmul(np.transpose(weight),x))/len(train_data))
                #print(avgbias)
        #print(total / itera)
        #print('Subgrad. SVM test error is:',1-(total / itera), "\nAt C of", C[c_index],"\\\\")
        print(weight)
        print(avgbias)
        #print(gammat)
        return [testret,1-(total / itera)]


import scipy
def dual_predict(w,x,y,r):
    #print( np.sign(np.matmul(np.transpose(w),x)))
    if y.flat[0] != np.sign(np.matmul(np.transpose(w),x)):
        if y.flat[0] == -1: #mult by -1
            x = np.multiply(x, -1)
        ryx = np.multiply(r,x)
        #print(ryx)
        w = np.add(w,ryx)
    return w

def dual_test(w,x,y,b):
    if y.flat[0] != np.sign(np.matmul(np.transpose(w), x) + b):
        return 0
    return 1
C = [100/873,500/873,700/873]
def dual_gamma_basic(gamma0,alpha,t):
    return gamma0/(1+(gamma0/alpha)*t)
def dual_gamma_2(gamma0,alpha,t):
    return gamma0/(1+t)
def dual(rate = 1, alpha=1,epoch= 100,c_index = 0, altg=False):
    with open("bank-note/train.csv", 'r') as train:
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            if xarr[len(xarr)-1]==0:
                xarr[len(xarr)-1] = -1
            train_arr.append(xarr)
            alp = [0] * (len(train_arr))


        x_array = np.array(train_arr)[:,:-1]
        y_array = np.array(train_arr)[:,-1]
        #print(y_array.shape)
        xmult = np.matmul(x_array, np.transpose(x_array))
        ymult = np.matmul(y_array, np.transpose(y_array))
        xymul = xmult* ymult
        def function(al):
            # sum =0
            # for i_f in range(0,len(train_arr)):
            #     x_i = np.array(train_arr[i_f][:len(train_arr[i_f]) - 1])
            #     y_i = np.array(train_arr[i_f][len(train_arr[i_f]) - 1])
            #     for j_f in range(0,len(train_arr)):
            #         x_j = np.array(train_arr[j_f][:len(train_arr[j_f]) - 1])
            #         y_j = np.array(train_arr[i_f][len(train_arr[j_f]) - 1])
            #         sum+= ((y_i*y_j*a[i_f]*a[j_f]*np.matmul(np.transpose(x_i),x_j))/2)
            #
            #     sum -= a[i_f]


            #print(xmult.shape)
            #print(ymult.shape)

            #print(xymul.shape)
            #at = np.transpose(a)
            a = np.array(al)
            axymult = np.matmul(np.transpose(a), xymul)
            output = np.matmul(axymult,a)
            #print(output)


            return (.5*output) - np.sum(a)
        bound = [(0.0,C[c_index])]*(len(train_arr))
        sum = 0


            #print (bound)
        #print(C[c_index])
        result = scipy.optimize.minimize(function,alp,bounds=tuple(bound))
        #print(result.x)
        for i in range(len(train_arr)):
            sum += alp[i] * y_array[i]
       # print(sum)
        w = np.multiply(result.x[0]*y_array[0],x_array[0])
        for i in range(1,len(train_arr)):
            if result.x[i] == 0.0:
                continue
            w += np.multiply(result.x[i]*y_array[i],x_array[i])
        print(w)
        bias  = 0
        for i in range(0,len(train_arr)):
            bias += (y_array[i]-np.matmul(np.transpose(w),x_array[i]))/len(train_arr)
        print("Bias:", bias, "\\\\")

        #debug: does it train ok?
        total =0
        itera = 0
        for train_data in train_arr:  # for xi,yi
            itera+=1
            x = np.array(train_data[:len(train_data) - 1])
            y = np.array(train_data[len(train_data) - 1])
            if y == 0.0:
                y = np.array(-1)
            total+= dual_test(w,x,y,bias)
        #print('DualTrain error is:',1-(total / itera), "\nAt C of", C[c_index],"\\\\")
        testret = 1-(total / itera)
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
                total+= dual_test(w,x,y,bias)
        print('dual SVM test error is:',1-(total / itera), "\nAt C of", C[c_index],"\\\\")
        return [testret,1-(total / itera)]
xguassglobal = None
def kernel_gaus(xi,xj,gamma):
    gaussian = math.exp(-1*np.linalg.norm(np.subtract(xi,xj))/gamma)
    #input()
    #print(gaussian)
    return gaussian
def predict_guass(w,x,y,b,xarr,yarr,a,gamma):
    sum2 = 0
    for i in range(len(xarr)):
        sum2+= (a[i]*yarr[i]*kernel_gaus(xarr[i],x,gamma))+b

    check = np.sign(sum2)
    if check == y:
        return 1
    return 0


def gauss(rate = 1, alpha=1,epoch= 100,c_index = 0, altg=False):
    with open("bank-note/train.csv", 'r') as train:
        train_arr = []
        weight = []
        for line in train:
            xarr = line.strip().split(',')
            for x in range(0,len(xarr)):
                 xarr[x]= float(xarr[x])
            if xarr[len(xarr)-1]==0:
                xarr[len(xarr)-1] = -1
            train_arr.append(xarr)
            alp = [0] * (len(train_arr))


        x_array = np.array(train_arr)[:,:-1]
        y_array = np.array(train_arr)[:,-1]
        #print(y_array.shape)
        xmult = np.empty((872,872))
        for i_f in range(0,len(train_arr)):
            for j_f in range(0, len(train_arr)):
                xmult[i_f][j_f] = kernel_gaus(x_array[i_f],x_array[j_f],rate)

        ymult = np.matmul(y_array, np.transpose(y_array))
        xymul = xmult* ymult
        #print(xmult)
        def function(al):
            # sum =0
            # for i_f in range(0,len(train_arr)):
            #     x_i = np.array(train_arr[i_f][:len(train_arr[i_f]) - 1])
            #     y_i = np.array(train_arr[i_f][len(train_arr[i_f]) - 1])
            #     for j_f in range(0,len(train_arr)):
            #         x_j = np.array(train_arr[j_f][:len(train_arr[j_f]) - 1])
            #         y_j = np.array(train_arr[i_f][len(train_arr[j_f]) - 1])
            #         sum+= ((y_i*y_j*a[i_f]*a[j_f]*np.matmul(np.transpose(x_i),x_j))/2)
            #
            #     sum -= a[i_f]


            #print(xmult.shape)
            #print(ymult.shape)

            #print(xymul.shape)
            #at = np.transpose(a)
            a = np.array(al)
            axymult = np.matmul(np.transpose(a), xymul)
            output = np.matmul(axymult,a)
            #print(output)


            return (.5*output) - np.sum(a)
        bound = [(0.0,C[c_index])]*(len(train_arr))
        sum = 0


            #print (bound)
        #print(C[c_index])
        result = scipy.optimize.minimize(function,alp,bounds=tuple(bound))
        #print(result.fun)
        #input()
        for i in range(len(train_arr)):
            sum += alp[i] * y_array[i]
       # print(sum)
        w = np.multiply(result.x[0]*y_array[0],x_array[0])
        for i in range(1,len(train_arr)):
            if result.x[i] == 0.0:
                continue
            w += np.multiply(result.x[i]*y_array[i],x_array[i])
        print("$", w, "$\\\\")
        bias  = 0
        for i in range(0,len(train_arr)):
            bias += (y_array[i]-np.matmul(np.transpose(w),x_array[i]))/len(train_arr)
        print("Bias:", bias, "\\\\")

        #debug: does it train ok?
        total =0
        itera = 0
        for train_data in train_arr:  # for xi,yi
            itera+=1
            x = np.array(train_data[:len(train_data) - 1])
            y = np.array(train_data[len(train_data) - 1])
            if y == 0.0:
                y = np.array(-1)
            total+= predict_guass(w,x,y,bias,x_array,y_array,result.x,rate)
        print('DualTrain error is:',1-(total / itera), "\nAt C of", C[c_index],"\\\\")
        testret = 1-(total / itera)
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
                total+= predict_guass(w,x,y,bias,x_array,y_array,result.x,rate)
        print('dual SVM test error is:',1-(total / itera), "\nAt C of", C[c_index],"\\\\")
        return [testret,1-(total / itera)]
if __name__ == '__main__':
    for i in range(0,3):
        b1 = basic(rate=.1, alpha=.3,c_index=i,altg=True)
        b2 = basic(rate=.1, alpha=.3,c_index=i,altg=False)
        print(abs(b1[0]-b2[0]),"\n",abs(b1[1]-b2[1]))
    print("DONE WITH PRIMAL SVM")
    for itee in range(0,3):
        b1 = dual(c_index=itee)
        b2 = dual(rate=.1, alpha=.3,c_index=i,altg=False)
        print(abs(b1[0]-b1[1]))
    print("DONE WITH DUAL SVM")
    gammas = [.1,.5,1,5,100]
    for j in range(0, 5):
        print("Gamma = ", gammas[j], "\\\\")
        for i in range(0,3):
                b1 = gauss(c_index=i,rate=gammas[j])
    print("DONE WITH GAUSSIAN SVM")
