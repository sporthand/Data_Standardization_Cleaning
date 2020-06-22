from Test01.utils import *
from imutils import paths
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from mx_imagenet_alexnet.pyimagesearch.preprocessing import ImageToArrayPreprocessor
from mx_imagenet_alexnet.pyimagesearch.preprocessing import AspectAwarePreprocessor
from mx_imagenet_alexnet.pyimagesearch.datasets import SimpleDatasetLoader
from keras.preprocessing.image import ImageDataGenerator

dataset_path = 'D:\\Porject001\\Datasets\\ImgDatas_03\\JHK_FER_2013_04.csv'
train_data_dir = 'D:\\Porject001\\Datasets\\ImgDatas_07\\Training'
validation_data_dir = "D:\\Porject001\\Datasets\\ImgDatas_07\\PublicTest"
# validation_data_dir = 'D:\\Porject001\\Datasets\\ImgDatas_06\\PublicTest'

# train_data_dir = 'D:\\Porject001\\Datasets\\CIFAR-10-images-master\\train'
# validation_data_dir = 'D:\\Porject001\\Datasets\\CIFAR-10-images-master\\test'






def data_augmentation():
    aug = ImageDataGenerator(#rescale = 1./255,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)
    return aug



def load_img_datasets(img_rows,img_cols,batch_size):

    # Augumenting Data with Flipping, Rotating, Zooming, Cropping, Varrying Color on data pictures



    # train_datagen = ImageDataGenerator(rescale = 1./255,
    #                                    rotation_range=20,
    #
    #                                    featurewise_center=False,
    #                                    featurewise_std_normalization=False,
    #                                    width_shift_range=0.1,
    #                                    shear_range= 0.2,
    #                                    zoom_range = 0.1,
    #                                    horizontal_flip=True,
    #                                    brightness_range=(0.5,1.5)
    #                                    )



    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       featurewise_center=False,
                                       featurewise_std_normalization=False,
                                       rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=.1,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale= 1./255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        color_mode= 'grayscale',
                                                        target_size=(img_rows,img_cols),
                                                        batch_size = batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True
                                                        )

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  color_mode='grayscale',
                                                                  target_size=(img_rows,img_cols),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  shuffle=True
                                                                  )

    return train_generator, validation_generator

def load_evaluate_dataset(img_rows, img_cols,batch_size,  private_test_path):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(private_test_path,
                                                                  color_mode='grayscale',
                                                                  target_size=(img_rows, img_cols),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  shuffle=True
                                                                  )
    return validation_generator

def get_data_process():


    data_gen = ImageDataGenerator( # to test it!
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True)
    return data_gen

def Split_Train_Test(num_labels):



    x_train, y_train, x_test, y_test = [], [], [], []
    df = pd.read_csv(dataset_path)

    for index, row in df.iterrows():
        val = row['pixels'].split(" ")
        try:
            if 'Training' in row['Usage']:
                x_train.append(np.array(val, 'float32'))
                y_train.append(row['emotion'])
            elif 'PublicTest' in row['Usage']:
                x_test.append(np.array(val, 'float32'))
                y_test.append(row['emotion'])
            # elif 'PrivateTest' in row['Usage']: # I add more testing data
            #     x_test.append(np.array(val, 'float32'))
            #     y_test.append(row['emotion'])

        except:
            print(f"error occured at index :{index} and row:{row}")

    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')

    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    y_train = np_utils.to_categorical(y_train, num_classes=num_labels)
    y_test = np_utils.to_categorical(y_test, num_classes=num_labels)

    x_train -= np.mean(x_train, axis=0)
    x_train /= np.std(x_train, axis=0)

    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)


    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)

    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)



    return x_train, y_train, x_test, y_test

def resize_crop_input_data(img_rows,img_cols):


    print ("---LOAINDG TRAINING AND PUBLIC TESTING!---")
    train_image_path = list(paths.list_images(train_data_dir))
    train_classNames = [pt.split(os.path.sep)[-2] for pt in train_image_path]
    train_classNames = [str(x) for x in np.unique(train_classNames)]  # reduce the redundant name of classifications

    test_image_path = list(paths.list_images(validation_data_dir))
    test_classNames = [pt.split(os.path.sep)[-2] for pt in test_image_path]
    test_classNames = [str(x) for x in np.unique(test_classNames)]

    aap = AspectAwarePreprocessor(img_rows, img_cols, inter=1)
    iap = ImageToArrayPreprocessor()


    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (train_data, train_labels) = sdl.load(train_image_path, verbose=500)
    (test_data, test_labels) = sdl.load(test_image_path, verbose= 500)
    train_data = train_data.astype("float")/255.0  # rescaling the color intensity of pixel
    test_data = test_data.astype("float")/255.0
    # print (f"train_data: {train_data}")
    train_labels = LabelBinarizer().fit_transform(train_labels)
    test_labels = LabelBinarizer().fit_transform(test_labels)
    print (f"train_data_shape: {train_data.shape}")
    print (f"test_data shape: {test_data.shape}")
    # print (f"type: {type(train_data)}")

    return train_data, test_data, train_labels, test_labels


def get_evaluate_test_data(img_rows, img_cols, private_test_path):
    print("---LOADING PRIVATE TESTING---")
    private_test_image_path = list(paths.list_images(private_test_path))
    private_test_classNames = [pt.split(os.path.sep)[-2] for pt in private_test_image_path]
    private_test_classNames = [str(x) for x in np.unique(private_test_classNames)]  # reduce the redundant name of classifications

    aap = AspectAwarePreprocessor(img_rows, img_cols)
    iap = ImageToArrayPreprocessor()

    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (private_test_data, private_test_labels) = sdl.load(private_test_image_path, verbose=500)
    private_test_data = private_test_data.astype("float")/255.0
    private_test_labels = LabelBinarizer().fit_transform(private_test_labels)

    return private_test_data, private_test_labels
# ###################################################<File Test Field>##################################################
if __name__ == "__main__":
    resize_crop_input_data(48,48)
    get_evaluate_test_data(48,48)





# def load_fer2013():
#     data = pd.read_csv(dataset_path)
#     pixels = data['pixels'].tolist()
#     width, height = img_cols, img_rows
#     faces = []
#     for pixel_sequence in pixels:
#         face = [int(pixel) for pixel in pixel_sequence.split(' ')]
#         face = np.asarray(face).reshape(width, height)
#         face = cv2.resize(face.astype('uint8'), image_size)
#         faces.append(face.astype('float32'))
#     faces = np.asarray(faces)
#     faces = np.expand_dims(faces, -1)
#     emotions = pd.get_dummies(data['emotion']).as_matrix()
#     return faces, emotions
#
#
# def preprocess_input(x, v2=True):
#     x = x.astype('float32')
#     x = x / 255.0
#     if v2:
#         x = x - 0.5
#         x = x * 2.0
#     return x


# load_fer2013()
