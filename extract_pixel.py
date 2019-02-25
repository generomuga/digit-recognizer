import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_header(height, width):
    #Get header for csv
    header = ''
    loop = int(height) * int(width)
    for i in range(loop):
        header = header+','+str(i)
    return header

#Prepare csv
global file_output
file_output = open('data\\data.csv', 'w')
file_output.write('label'+str(get_header(28, 28))+'\n')

def get_image_size(image):
    #Get image width and height
    return image.shape

def get_pixel_value(filepath, filename):
    #Status
    print ('Getting image pixels...',filepath)

    #Read the image
    image = cv2.imread(filepath)

    #Get image shape
    height, width, channels = get_image_size(image)
    #print ('Height: '+str(height)+' Width: '+str(width)+' Channels: '+str(channels))

    #Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray_image, cmap='gray')
    #plt.show()

    #Save to list
    list_pixel_value = []
    for x in range(width):
        for y in range(height):
            pixel_value = gray_image[x,y]
            list_pixel_value.append(pixel_value)

    #Graph
    #array = np.array(list_pixel_value)
    #visualize(array, height, width)

    #Save to csv
    write_csv('data\\data.csv', '-1', list_pixel_value)

def write_csv(filename, label, list_value):
    #Join array
    join_item = ', '.join(str(value) for value in list_value)
    file_output.write(label+','+join_item+'\n')

def visualize(array, height, width):
    plt.imshow(np.array(array.reshape(height, width)), cmap='gray')
    plt.show()

if __name__ == '__main__':
    dir = 'image\\'

    try:
        for item in os.listdir(dir):
            get_pixel_value(dir+item, item)
        print ('Done!')
    except:
        print ('Please check your images!')
