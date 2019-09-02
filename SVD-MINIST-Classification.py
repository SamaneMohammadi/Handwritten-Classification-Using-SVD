import numpy as np
from sklearn import datasets
import pandas as pd



#about data:Attribute Information: 8x8 image of integer pixels in the range 0..16.Number of Instances: 5620,Number of Attributes: 64
# load data MINIST from dataset sklearn
data = datasets.load_digits()
data_frame = pd.DataFrame(data['data'])
data_frame['label'] = data['target']
#print(data)



#find data for zero and label then transpose data and remove label from data


data_0 = data_frame[data_frame['label'] == 0][0:175].T
labels_0 = data_0.iloc[-1]
data_0 = data_0[0:-1]

#find data for one and label then transpose data and remove label from data
data_1 = data_frame[data_frame['label'] == 1][0:175].T
labels_1 = data_1.iloc[-1]
data_1 = data_1[0:-1]

#find data for two and label then transpose data and remove label from data
data_2 = data_frame[data_frame['label'] == 2][0:175].T
two_labels = data_2.iloc[-1]
data_2 = data_2[0:-1]



#selecte 150 data for train and 25 data for test
train_0 = data_0[data_0.columns[0:150]]
test_0 = data_0[data_0.columns[150:]]

train_1 = data_1[data_1.columns[0:150]]
test_1 = data_1[data_1.columns[150:]]

train_2 = data_2[data_2.columns[0:150]]
test_2 = data_2[data_2.columns[150:]]


#use svd for decomposition train matrix
u0, s0, v0 = np.linalg.svd(train_0)
u1, s1, v1 = np.linalg.svd(train_1)
u2, s2, v2 = np.linalg.svd(train_2)



#the func is for classify digit and find the closet value for imageVector
# |(I − Uk UkT )z|2 /| z|2 in least squares problem

def svd_classify(imageVector):

    Diff_0 = np.linalg.norm((np.identity(len(u0))[:d,:] - (u0[:d,:]*(u0[:,:d].T))).dot(imageVector),ord=2)/np.linalg.norm(imageVector)
    Diff_1 = np.linalg.norm((np.identity(len(u1))[:d,:] - (u1[:d,:]*(u1[:,:d].T))).dot(imageVector),ord=2)/np.linalg.norm(imageVector)
    Diff_2 = np.linalg.norm((np.identity(len(u2))[:d,:] - (u2[:d,:]*(u2[:,:d].T))).dot(imageVector),ord=2)/np.linalg.norm(imageVector)
    values = [Diff_0, Diff_1, Diff_2]
    #print(values)

    return values.index(min(values))


#We use svd classifier diff value for d
for i in range(1,65,10):
    print('Using D = ', i)
    d = i
    predict_0 = []
    predict_1 = []
    predict_2 = []
    threes_pred = []
    for i in range(len(test_1.columns)):
        predict_0.append(svd_classify(test_0.T.iloc[i]))
        predict_1.append(svd_classify(test_1.T.iloc[i]))
        predict_2.append(svd_classify(test_2.T.iloc[i]))
    correct_pre_0 = predict_0.count(0)/1.0/len(predict_0)
    correct_pre_1 = predict_1.count(1)/1.0/len(predict_1)
    correct_pre_2 = predict_2.count(2)/1.0/len(predict_2)


    print ("Correct Prediction Percentages 0 : ",correct_pre_0)
    print ("Correct Prediction Percentages 1 : ",correct_pre_1)
    print ("Correct Prediction Percentages 2 : ",correct_pre_2)
    print ("Accuracy : ",(correct_pre_0 + correct_pre_0 + correct_pre_0)/3.0)
    print ('-----------------------------------------------------')


