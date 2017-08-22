
from model_car.model.model import *
from model_car.data_analysis.data_parsing.get_data_from_bag_files import *

import numpy as np
from numpy import *
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import cv2

weights_file_path = sys.argv[1]
hdf5_filename = sys.argv[2]
version = 'version 1b'
solver_file_path = 'z2_color_' + version
#weights_file_mode = 'most recent' #'this one' #None #'most recent' #'this one'  #None #'most recent'
#weights_file_path = opjD('/home/bdd/git/model_car/model_car/model/z2_color_tf.npy') #opjD('z2_color_long_train_21_Jan2017') #None #opjh('kzpy3/caf6/z2_color/z2_color.caffemodel') #None #'/home/karlzipser/Desktop/z2_color' # None #opjD('z2_color')
weights_file_path = opjD(weights_file_path)
model = get_model(version, phase='train')
model = load_model_weight(model, weights_file_path)
model.compile(loss = 'mean_squared_error',
                              optimizer = optimizers.SGD(lr = 0.01, momentum = 0.001, decay = 0.000001, nesterov = True),
                              metrics=['accuracy'])
model.summary()

ZED_input = {}
meta_input = {}
#hdf5_filename = '/home/bdd/Desktop/output_hdf5/hdf5/runs/08.hdf5' #runs_folder = '~/Desktop/tmp/hdf5/runs'
hdf5_content = h5py.File(hdf5_filename, 'r')
print hdf5_filename
kk = hdf5_content.keys()
dt=33
while True:
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
            img[:,:,0] = hdf5_content[k]['ZED_input'][0,i+8,:,:].copy()
            img[:,:,1] = hdf5_content[k]['ZED_input'][0,i+4,:,:].copy()
            img[:,:,2] = hdf5_content[k]['ZED_input'][0,i,:,:].copy()
            img = z2o(img)
            # print(pre_steer)
            # bar_len = int(20+pre_steer[9])
            # img = cv2.rectangle(img, (100, 20), (101, 60), (48, 156, 245), -1)
            # print('pre_steer:{}, act_steer:{},  pre_motor:{}, act_motor:{}'.format(100-pre_steer, 100-actual_steer, pre_motor, actual_motor))
            
            
            cv2.rectangle(img, (200, 40), (200 - 2 * int(pre_steer) + 49 * 2, 60), (0, 0, 255), -1)
            cv2.rectangle(img, (200, 70), (200 - 2 * int(actual_steer) + 49 * 2, 90), (0, 0, 255), -1)
            cv2.rectangle(img, (500, 120), (480, 100 - int(pre_motor) + 49), (0, 255, 255), -1)
            cv2.rectangle(img, (510, 120), (530, 100 - int(actual_motor) + 49), (0, 255, 255), -1)

            cv2.line(img, (200, 40), (200, 90), (1,1,1), 1)
            cv2.line(img, (480, 120), (530, 120), (10,10,10), 1)

            # img = cv2.putText(img, 'Prediction_Steer', )
            
            
            cv2.imshow('Prediction_Steer',img)
            if cv2.waitKey(dt) & 0xFF == ord('q'):
                pass
            
            #plt.imshow(img)
            # pause(10)





