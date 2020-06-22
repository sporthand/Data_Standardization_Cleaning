from PIL import Image
import numpy as np
import sys
import os
import csv

base_directory = 'D:\\Porject001\\Datasets\\ImgDatas_03\\'
folder_name = "PublicTest"
file_name = "JHK_FER_2013_03_02.csv"


# default format can be changed as needed
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList(base_directory+folder_name)
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

if (folder_name == "Training"):
    with open(base_directory+file_name, "a") as f:
        f.write(f'emotion,pixels,Usage')
        f.write("\n")

for file in myFileList:
    print(file)
    img_file = Image.open(file)

    category = os.path.basename(os.path.dirname(file))
    # img_file.show()

    # get original image parameters...
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

    # Make image Greyscale
    img_grey = img_file.convert('L')
    # img_grey.save('result.png')
    # img_grey.show()


    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
    value = value.flatten()
    print(value)
    # value = str(value).lstrip('[').rstrip(']')
    with open(base_directory+file_name, "a") as f:
        f.write(f'{label_names.index(category)},{" ".join([str(pixel) for pixel in value.reshape(2304)])},{folder_name}')

        f.write("\n")
