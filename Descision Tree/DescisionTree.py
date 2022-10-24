with open("../Storage/concrete/train.csv" , 'r') as train:
    for line in train:
        xarr = line.strip().split(',')
        #do stuff