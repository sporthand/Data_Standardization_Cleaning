from PIL import Image
import cv2
import os
from mtcnn.mtcnn import MTCNN

height_start_crop = 0.15
height_end_crop = 0.80

width_start_crop = 0.25
width_end_crop = 0.75

base_directory = "D:\\Porject001\\Test02\\raw_data\\"
generated_path = "D:\\Porject001\\Test02\\Generated_Data03\\"
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_images_folder(base_directory, generated_path,model, format = '.png' ):
    #print(myDir)
    for root, dirs, files in os.walk(base_directory, topdown = False):
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root,name)
                dirname = root.split(os.path.sep)[-1]
                img_file= cv2.imread(full_name)
                faces = model.detect_faces(img_file)
                for i in range(len(faces)):
                    start_col, start_row, width, height = faces[i]['box']
                    end_col, end_row = start_col + width, start_row + height

                    new_img_file = img_file[start_row:end_row,start_col: end_col]
                    dirpath = os.path.join(generated_path, dirname)
                    new_name = os.path.join(dirpath, str(i)+name )
                    cv2.imwrite(new_name, new_img_file)
                    if not os.path.exists(dirpath):
                        os.system('mkdir {}'.format(dirpath))

                    print(img_file.shape)
                    print(full_name)
                    print(new_img_file.shape)
                    print(new_name)
                # implement here


if __name__ == "__main__":
    model = MTCNN()

    load_images_folder(base_directory,generated_path, model)
