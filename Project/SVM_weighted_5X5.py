#SVM Weighted 5X5
#SVM Classificaton Method

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


#function to perform convolution
def convolve2D(image, filter):
  fX, fY = filter.shape # Get filter dimensions
  fNby2 = (fX//2) 
  n = 28
  nn = n - (fNby2 *2) #new dimension of the reduced image size
  newImage = np.zeros((nn,nn)) #empty new 2D imange
  for i in range(0,nn):
    for j in range(0,nn):
      newImage[i][j] = np.sum(image[i:i+fX, j:j+fY]*filter)//25
  return newImage

#Read Data from CSV
train = pd.read_csv("train.csv")
X = train.drop('label',axis=1)
Y = train['label']

#Create Filter for convolution
filter = np.array([[1,1,1,1,1],
          [1,2,2,2,1],
          [1,2,3,2,1],
          [1,2,2,2,1],
          [1,1,1,1,1]])

#convert from dataframe to numpy array
X = X.to_numpy()

#new array with reduced number of features to store the small size images
sX = np.empty((0,576), int)


ss = 42000 

#Perform convolve on all images
for img in X[0:ss,:]:
  img2D = np.reshape(img, (28,28))
  nImg = convolve2D(img2D,filter)
  nImg1D = np.reshape(nImg, (-1,576))
  sX = np.append(sX, nImg1D, axis=0)

Y = Y.to_numpy()
sY = Y[0:ss]

# train and test model
sXTrain, sXTest, yTrain, yTest = train_test_split(sX,sY,test_size=0.2,random_state=0)
clf = SVC()
clf.fit(sXTrain, yTrain)

#Printing our score and creating predictions
print(clf.score(sXTest, yTest))
prediction = clf.predict(sXTest)

#Reading our sample submission file and upadting it
submissionFile=pd.read_csv('sample_submission.csv')
submissionFile['Label'] = prediction
submissionFile.to_csv('SVM5x5.csv', index=False)