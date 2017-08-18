
from model_car.model.model import *
from model_car.data_analysis.data_parsing.get_data_from_bag_files import *

import numpy as np
from numpy import *
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import cv2

version = 'version 1b'
solver_file_path = 'z2_color_' + version
#weights_file_mode = 'most recent' #'this one' #None #'most recent' #'this one'  #None #'most recent'
#weights_file_path = opjD('/home/bdd/git/model_car/model_car/model/z2_color_tf.npy') #opjD('z2_color_long_train_21_Jan2017') #None #opjh('kzpy3/caf6/z2_color/z2_color.caffemodel') #None #'/home/karlzipser/Desktop/z2_color' # None #opjD('z2_color')
weights_file_path = opjD('/home/eralien/output_hdf5/z2_color_version_1b_final_run131140.hdf5')
model = get_model(version, phase='train')
model = load_model_weight(model, weights_file_path)
model.compile(loss = 'mean_squared_error',
                              optimizer = optimizers.SGD(lr = 0.01, momentum = 0.001, decay = 0.000001, nesterov = True),
                              metrics=['accuracy'])
# model.summary()

ZED_input = {}
meta_input = {}
hdf5_filename = '/media/eralien/4TB_rosbag1/output_hdf5/08Aug/hdf5/runs/20.hdf5' #runs_folder = '~/Desktop/tmp/hdf5/runs'
hdf5_content = h5py.File(hdf5_filename, 'r')
print hdf5_filename
kk = hdf5_content.keys()
for k in kk:
    if "True" in k:
        continue
    ZED_input = hdf5_content[k]['ZED_input'][:]/255.-0.5
    meta_input = hdf5_content[k]['meta_input'][:]
    # print(ZED_input[0, 0, :, :])
    prediction = model.predict_on_batch({'ZED_input': ZED_input, 'meta_input': meta_input})
    pre_steer = 100 * prediction[0, 9]
    pre_motor = 100 * prediction[0, 19]
    # print(pre_steer)
    actual_steer = 100 * hdf5_content[k]['steer_motor_target_data'][0,9]
    actual_motor = 100 * hdf5_content[k]['steer_motor_target_data'][0,19]
    # print(actual_motor, actual_steer)

    data_img_shape = np.shape(hdf5_content[k]['ZED_input'])
    #print(data_img_shape)
    img = np.zeros((data_img_shape[2], data_img_shape[3], 3))
    #print(img.shape)
    # r = 3
    for i in range(2):
        # img[:,:data_img_shape[3],0] = hdf5_content[k]['ZED_input'][0,r+i,:,:].copy()
        # img[:,:data_img_shape[3],1] = hdf5_content[k]['ZED_input'][0,r+i+6,:,:].copy()
        # img[:,:data_img_shape[3],2] = hdf5_content[k]['ZED_input'][0,r+i+12,:,:].copy()
        img[:,:,0] = hdf5_content[k]['ZED_input'][0,i,:,:].copy()
        img[:,:,1] = hdf5_content[k]['ZED_input'][0,i+4,:,:].copy()
        img[:,:,2] = hdf5_content[k]['ZED_input'][0,i+8,:,:].copy()
        img = z2o(img)
        # print(pre_steer)
        # bar_len = int(20+pre_steer[9])
        # img = cv2.rectangle(img, (100, 20), (101, 60), (48, 156, 245), -1)
        img = cv2.rectangle(img, (200, 40), (200 - 2 * int(pre_steer) + 49 * 2, 60), (50, 213, 213), -1)
        img = cv2.rectangle(img, (200, 70), (200 - 2 * int(actual_steer) + 49 * 2, 90), (50, 70, 200), -1)
        img = cv2.rectangle(img, (500, 120), (480, 100 - int(pre_motor) + 49), (50, 213, 213), -1)
        img = cv2.rectangle(img, (510, 120), (530, 100 - int(actual_motor) + 49), (50, 70, 200), -1)

        img = cv2.line(img, (200, 40), (200, 90), (1,1,1), 1)
        img = cv2.line(img, (480, 120), (530, 120), (10,10,10), 1)

        # img = cv2.putText(img, 'Prediction_Steer', )

        plt.imshow(img)
        pause(0.1)
        # pause(10)





