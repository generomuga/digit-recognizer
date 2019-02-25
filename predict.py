import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def classify_image(filename):
    #Load data
    data = pd.read_csv(filename).as_matrix()

    #Get train label
    train_label = data[0:1000, 0]
    #Get train feature (pixels)
    train_feature = data[0:1000, 1:]

    #Get test label
    test_label = data[0:100, 0]
    #Get test feature
    test_feature = data[0:100, 1:]

    #Get to predict
    predict_label, predict_feature = read_to_predict('data\\data.csv')

    #Graph
    visualize(predict_feature[0], 28, 28)

    #Set classifier
    clf = svm.SVC(C=1.0, cache_size=600, kernel='linear', gamma=0.001, class_weight='balanced')
    #Fit data to model
    clf.fit(train_feature, train_label)
    #predict
    print ('Predicted digit: ', str(clf.predict([predict_feature[0]])))

def visualize(array, height, width):
    plt.imshow(np.array(array.reshape(height, width)), cmap='gray')
    plt.show()

def read_to_predict(filename):
    data = pd.read_csv(filename).as_matrix()
    #Get test label
    test_label = data[0:, 0]
    #Get test feature
    test_feature = data[0:, 1:]

    return test_label, test_feature

if  __name__ == '__main__':
    classify_image('data\\train.csv')
