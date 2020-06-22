from mx_imagenet_alexnet.pyimagesearch.preprocessing import ImageToArrayPreprocessor
from mx_imagenet_alexnet.pyimagesearch.preprocessing import AspectAwarePreprocessor
from mx_imagenet_alexnet.pyimagesearch.datasets import SimpleDatasetLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import sys

base_directory = "D:\\Porject001\\Test02\\Generated_Data03\\"
generated_path = "D:\\Porject001\\Test02\\Generated_Data04\\"
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_images_folder(base_directory, generated_path, format = '.png' ):
    file_list = []
    aap = AspectAwarePreprocessor(48, 48)


    #print(myDir)
    for root, dirs, files in os.walk(base_directory, topdown = False):
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root,name)
                dirname = root.split(os.path.sep)[-1]
                img_file= cv2.imread(full_name)
                img_file = aap.preprocess(img_file)
                file_list.append(full_name)
                dirpath = os.path.join(generated_path, dirname)
                cv2.imwrite(os.path.join(dirpath, name), img_file)
                if not os.path.exists(dirpath):
                    os.system('mkdir {}'.format(dirpath))




# def wrtie_images_folder(images,generated_path):
#     for image = cv2.imwrite""


if __name__ == "__main__":

    load_images_folder(base_directory,generated_path )
    # print (original_imgs)

    # plt.imshow(img)
    # plt.show()


