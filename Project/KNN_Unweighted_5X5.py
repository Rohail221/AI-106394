#KNN Unweighted 5X5
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import math
train = pd.read_csv("train.csv")  # reading data from dataset ,
#dataset from kaggle digit recognizer competition
X = train.drop('label',axis=1)
Y = train['label']
# the filter for convolution 5*5 (unweighted)

filter = np.array([[1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1]])

X = X.to_numpy()  #conversion of datafrom to numpy array
print(X.shape)

#that new array which is reduced
size = np.empty((0,576), int)

#creating a function which can perform convolution 
def convolution(image, filter):
  fX, fY = filter.shape 
# for getting filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2)
    #mentioned new dimensions of image which has been reduced
newImage = np.zeros((nn,nn))
#take new and empty 2D image
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

subset = 500 
#subset size for dry runs change to 42000 to run on whole data
#convolving all images in below code
for img in X[0:subset,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolution(img2D,filter)
 
  nImg1D = np.reshape(nImg, (-1,576))
 
  sX= np.append(sX, nImg1D, axis=0) #size

Y = Y.to_numpy()
sY = Y[0:subset]
# print(sY)
print(sY.shape)
print(sX.shape)
#test and train your data here 
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
print(sXTest.shape,", ",yTest.shape)
print(sXTrain.shape,", ",yTrain.shape)
#K nearest neighbours algorithm technique starting from here 
yourdata = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
yourdata.fit(sXTrain,yTrain)
Y_pred = yourdata.predict(sXTest)
print(accuracy_score(yTest,Y_pred))
