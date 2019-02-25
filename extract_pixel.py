import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_image_size(image):
    #Get image width and height
    return image.shape

def get_pixel_value(filename):
    #Read the image
    image = cv2.imread(filename)

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

    #Save to csv
    write_csv('2', list_pixel_value, 'data\\data.csv', height, width)

def get_header(height, width):
    #Get header for csv
    header = ''
    loop = int(height) * int(width)
    for i in range(loop):
        header = header+','+str(i)
    return header

def write_csv(label, list_value, filename, height, width):
    #Join array
    join_item = ', '.join(str(value) for value in list_value)
    #Write in csv format
    file_output = open(filename, 'w')
    file_output.write('label'+str(get_header(height, width))+'\n')
    file_output.write(label+','+join_item+'\n')

if __name__ == '__main__':
    get_pixel_value('image/test-2.png')
