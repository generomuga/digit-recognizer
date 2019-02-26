import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_rgb2gray(filename):
    #Load data
    data = pd.read_csv(filename).as_matrix()

    #Get train feature (pixels)
    feature = data[0:, 0:]

    for i in range(0, 10):
        gray_image = np.array(feature[i]).reshape((28,28))
        #plt.imshow(gray_image, cmap='gray')
        #plt.show()

        cv2.imwrite('image\\t'+str(i)+'.png', gray_image)

if __name__ == '__main__':
    convert_rgb2gray('data\\test.csv')
