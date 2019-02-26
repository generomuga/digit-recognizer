import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

def classify_image(filename):
    #Load data
    data = pd.read_csv(filename).as_matrix()

    nsample = len(data)
    #print (nsample)

    #Get train label
    label = data[0:2000, 0]
    #Get train feature (pixels)
    feature = data[0:2000, 1:]

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, random_state=42)

    #Get to predict
    predict_label, predict_feature = read_to_predict('data\\data.csv')

    #Graph
    index_img = 0
    visualize(predict_feature[index_img], 28, 28)

    #Set classifier
    clf = svm.SVC(C=1.0, cache_size=600, kernel='linear', gamma=0.0001, class_weight='balanced')
    #clf = svm.SVC(gamma=0.0001)

    #Fit data to model
    clf.fit(X_train, y_train)

    #predict
    print ('Predicted digit: ', str(clf.predict([predict_feature[index_img]])))
    print ('Accuracy', clf.score(X_test, y_test))

def visualize(array, height, width):
    plt.imshow(np.array(array.reshape(height, width)))
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
