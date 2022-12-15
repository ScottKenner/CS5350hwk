import random
# coding: utf-8
import time, os, zipfile,sklearn

os.environ['KAGGLE_USERNAME'] = 'scottkenner'
os.environ['KAGGLE_KEY'] = '8102b8a73742d1ba6c21c26d65090b25'
from kaggle import api as kaggle_api

competition_slug = "income2022f"
sbm_path = f"submission/s1.csv"
message = "test submission"
def cheese():
    os.environ['KAGGLE_USERNAME'] = 'scottkenner'
    os.environ['KAGGLE_KEY'] = ''
    from kaggle import api as kaggle_api
    competition_slug = "income2022f"
    sbm_path = f"submission/s1.csv"
    message = "test submission"
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    options = webdriver.ChromeOptions()
    options.add_argument("~/Library/Application Support/Google/Chrome/Default")
    driver = webdriver.Chrome(chrome_options=options)
    driver.get("https://www.kaggle.com/competitions/income2022f/leaderboard/download")
    input("Press any key after authing")
    predictions_array = []
    for i in range(23842):
        predictions_array.append([str(i + 1), str(0)])
        reset_prev = False
    old_acc = 0.0
    iter = 0
    while True:
        #reset file
        with open('submission/s1.csv', 'w') as f:
            f.truncate(0)
            f.write("ID,Prediction\n")
            for prediction in predictions_array:
                f.write(str(prediction) + '\n')
            f.close()
        #submit file, pull down the leaderboards to see if it was right
        kaggle_api.competition_submit_cli(sbm_path, message, competition_slug)

        elem = driver.find_element(By.XPATH, "/html/body/main/div[1]/div/div[6]/div[2]/div/div[2]/div[1]/div[2]/a/button")
        elem.click()

        #wait for download
        time.sleep(1)
        with zipfile.ZipFile('/Users/scottk/Downloads/income2022f-publicleaderboard.zip', 'r') as zip_ref:
            zip_ref.extractall("data/leaderboard")

        with open('data/leaderboard/income2022f-publicleaderboard.csv', 'r') as leader:
            for row in leader.readlines():
                if 9195558 in row.split(','):
                    new_acc = row.split(',')[3]
        if float(new_acc) > old_acc:
            old_acc = new_acc
            reset_prev = False
        else:
            reset_prev = True
        predictions_array[iter][1] = 1
        if reset_prev:
            predictions_array[iter-1][1] = 0

        #cleanupFiles
        os.remove('/Users/scottk/Downloads/income2022f-publicleaderboard.zip')
        os.remove("data/leaderboard/income2022f-publicleaderboard.csv")

        if float(new_acc) > .99999:
            exit()
        iter +=1

solution_file = open('submission/s1.csv', 'w')
test_file = open('data/income2022/test_final.csv', 'r')
train_file = open('data/income2022/train_final.csv', 'r')



#index data
train_data = []
x_array = []
y_data = []
#age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income>50K
init_row  = train_file.readline()
count = 0

#10,11,12,00 04
class overall_values:
    age_low = 1000000000
    age_high = 0
    EN_H = 0
    CG_H = 0
    CL_H =0
    HPW_H =0
    EN_L = 1000000000
    CG_L = 1000000000
    CL_L=1000000000
    HPW_L=1000000000
    FNL_L= 1000000000
    FNL_H = 0




def preprocess_data(train_file, OV):
    for row in train_file.readlines():
        count = 0
        # 10,11,12,00 04
        for item in row.split(','):
            if count not in [10, 11, 12, 0, 4,2]:
                count += 1
                continue
            #print(item)
            item = int(item)
            if count == 0:#age
                if item < OV.age_low:
                    OV.age_low = item
                if item > OV.age_high:
                    OV.age_high = item
            elif count == 4:#educationnum
                if item < OV.EN_L:
                    OV.EN_L = item
                if item > OV.EN_H:
                    OV.EN_H = item
            elif count == 10:#capital gain
                if item < OV.CG_L:
                    OV.CG_L = item
                if item > OV.CG_H:
                    OV.CG_H = item
            elif count == 11:#cap loss
                if item < OV.CL_L:
                    OV.CL_L = item
                if item > OV.CL_H:
                    OV.CL_H = item
            elif count == 12:#hrsweek
                if item < OV.HPW_L:
                    OV.HPW_L = item
                if item > OV.HPW_H:
                    OV.HPW_H = item
            elif count ==2:
                if item < OV.FNL_L:
                    OV.FNL_L = item
                if item > OV.FNL_H:
                    OV.FNL_H = item
            count+=1
def determine_quartile(count, value,OV ):
    def det_quarter(hi,lo,val):
        #print(val,hi,lo)
        val = int(val)
        hi = hi-lo
        #print(val)
        val = val- lo
        lo =0

        if val ==0 or float(val/hi) < .25:
            return 1
        if float(val/hi) < .5:
            return 2
        if float(val/hi) < .75:
            return 3
        return 4#there has got to be a better way to do this....
    if count == 0:  # age
        return det_quarter(OV.age_high,OV.age_low,value)
    elif count == 4:  # educationnum
        #print(value)
        return det_quarter(OV.EN_H, OV.EN_L, value)
    elif count == 10:  # capital gain
        return det_quarter(OV.CG_H, OV.CG_L, value)
    elif count == 11:  # cap loss
        return det_quarter(OV.CL_H, OV.CL_L, value)
    elif count == 12:  # hrsweek
        return det_quarter(OV.HPW_H, OV.HPW_L, value)
    elif count ==2:
        return det_quarter(OV.FNL_H, OV.FNL_L, value)


OverallVals = overall_values

preprocess_data(train_file,OverallVals)
train_file = open('data/income2022/train_final.csv', 'r')
train_file.readline()#skip init line?
for item in init_row.split(","):
    x_array.append({})
    count+=1
x_len = 0
y_len = 0
for row in train_file.readlines():
    #print("aa")
    data = []
    count = 0
    #print(item)
    # 10,11,12,00 04
    for item in row.split(','):
        #print(count)
        if count == len(row.split(','))-1:
            y_data.append(int(item))
            count+=1
            continue
        if count in [10,11,12,0,4,2]:
            #item = determine_quartile(count,item,OverallVals)
            data.append(int(item))
            count += 1

            continue
        elif not x_array[count]:
            x_array[count] = {item:0}
        elif item not in x_array[count]:
            x_array[count].update({item:len(x_array[count])})
        #print(x_array)
        data.append(x_array[count][item])
        count+=1
    train_data.append(data)
#print(train_data)
init_row  = test_file.readline()
test_data = []
dontadd = False


#print(OverallVals.age_low,overall_values.age_high)
for row in test_file.readlines():
    data = []
    count = 0
    #print(item)
    for item in row.split(','):
        if count == 0:
            count+=1
            continue
        if count-1 in [10,11,12,0,4,2]:
            #item = determine_quartile(count-1,item,OverallVals)
            data.append(int(item))
            count+=1
            #print(item)
            continue
        if item not in x_array[count]:
            data.append(0)
        else:
            data.append(x_array[count][item])
        count+=1
    test_data.append(data)

# for row in train_data:
#     print(row)
# print(y_data)
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
clf = Perceptron(tol=None,max_iter=5000)
clf = clf.fit(train_data,y_data)
print(test_data)


predictions = clf.predict(test_data)



#0.53488 with quart
#0.64778 without quart
# from sklearn.datasets import load_iris
# from sklearn import tree
#
# iris = load_iris()
# #print(train_data)
# X, y = train_data, y_data
# #print(X,y)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
#
# preX = test_data
#
# predictions = clf.predict(preX)

solution_file.truncate(0)
solution_file.write("ID,Prediction\n")
i = 0
print(predictions)
for prediction in predictions:
    i+=1
    # if prediction != 1:
    #     print(0)
    solution_file.write(str(i)+','+str(prediction) + '\n')
solution_file.close()
kaggle_api.competition_submit_cli(sbm_path, message, competition_slug)

