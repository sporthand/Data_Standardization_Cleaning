from  Test01.utils import *
import datetime

from Test01.Create_Sparsity_Masks import create_sparsity_masks

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=10,
    inter_op_parallelism_threads=10
)
# tf.Session(graph=tf.get_default_graph(), config=session_conf)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.get_session(sess)


# ####################################################<Local Parameters!>#############################################
START = datetime.datetime.now()

batch_size = 64
num_epochs_1 = 10000
num_epochs_2 = 1
num_epochs_final = 1
num_classes = 7
num_patience = 50
base_path = 'D:\\Porject001\\Trained_Models\\'
num_rows, num_cols = 48, 48
input_shape = (num_rows,num_cols,1)
model_names = base_path + 'fer_{epoch:02d}-{val_accuracy:.4f}-{val_loss:.4f}.h5'
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss',
                                   verbose = 1,
                                   save_best_only=True,
                                   save_weights_only= False # True For Mixnet Testing
                                   )
early_stop = EarlyStopping('val_loss', patience = num_patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor= 0.1, patience= int(num_patience/4),verbose = 1)#, min_lr=10e-7)

callbacks = [early_stop, reduce_lr]
final_callbacks = [early_stop, reduce_lr, model_checkpoint]
# ####################################################################################################################

# ###########################################<For Actual Image File!>#################################################
# trainX, testX, trainY, testY = resize_crop_input_data(img_rows=num_rows, img_cols=num_cols)
# aug = data_augmentation()
train_generator,validation_generator = load_img_datasets(num_rows,num_cols,batch_size)

# ####################################################################################################################


# ########################################<model parameters/compilation>##############################################
model = mini_XCEPTION((input_shape), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
# ####################################################################################################################
# #######################################<For Actual Image Files!>####################################################
nb_train_sample = train_generator.n//train_generator.batch_size
nb_test_sample = validation_generator.n//validation_generator.batch_size

history = model.fit_generator(train_generator,
                              shuffle = True,
                              steps_per_epoch= nb_train_sample,
                              epochs = num_epochs_1,
                              verbose= 1,
                              callbacks = final_callbacks,
                              validation_data=validation_generator,
                              validation_steps=nb_test_sample
                              )
# ####################################################################################################################

END = datetime.datetime.now()

# #############################################<Plotting Graph>#######################################################
print ('\nStart Time:', START )
print ('End Time: ',END)
print ('Training Time Spent: ',END-START)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# ####################################################################################################################



# ############################################<Create Json from Neha!>################################################
# fer_json = model.to_json()
# with open(base_path+ "fer.json01", "w") as json_file:
#     json_file.write(fer_json)
# model.save_weights(base_path+"fer01.h5")
# ####################################################################################################################