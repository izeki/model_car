import numpy as np
from model_car.vis import *
from model_car.model.predict import *
import cv2
blue = [0,0,0.8]
blue_green = [0,0.5,0.5]

def training_data_generator(version,data,flip,show_data,camera_dropout):
    if version == 'version 1':
        return training_data_generator_1(data,flip,show_data,camera_dropout)
    if version == 'version 1b':
        return training_data_generator_1b(data,flip,show_data,camera_dropout)
    if version == 'squeeze_net':
        return training_data_generator_1b(data,flip,show_data,camera_dropout)
    if version == 'version 2':
        return training_data_generator_2(data,flip,show_data,camera_dropout)
    if version == 'version z3':
        return training_data_generator_z3(data,flip,show_data,camera_dropout)
    assert(False)
    
def visualize_data_model_version(version, data_set, flip):
	if version == 'version 1':
		return visualize_data_model_version_1(data_set, flip)
	if version == 'version 1b':
		return visualize_data_model_version_1b(data_set, flip)
	if version == 'version 2':
		return visualize_data_model_version_2(data_set,flip)
	assert(False)    

def training_data_generator_1(data,flip,show_data=False,camera_dropout=False):
    result = {}
    x_train = {}
    y_train = {}
    if 'left' in data:
        if len(data['left']) >= 10:
            if type(data['left'][0]) == np.ndarray:
                target_data = data['steer'][:10]
                target_data += data['motor'][:10]
                #mi(data['left'][0][:,:,:],'left')
                #mi(data['right'][0][:,:,:],'right')
                _, _, im_h, im_w = data['left'][0].shape
                ZED_input = np.zeros((1, 12, im_h, im_w))
                meta_input = np.zeros((1,6, 14, 26))
                steer_motor_target_data = np.zeros((1,20))                 
                
                if not flip:
                    ZED_input[0,0,:,:] = data['left'][0][:,:,0]
                    ZED_input[0,1,:,:] = data['left'][1][:,:,0]
                    ZED_input[0,2,:,:] = data['right'][0][:,:,0]
                    ZED_input[0,3,:,:] = data['right'][1][:,:,0]
                    ZED_input[0,4,:,:] = data['left'][0][:,:,1]
                    ZED_input[0,5,:,:] = data['left'][1][:,:,1]
                    ZED_input[0,6,:,:] = data['right'][0][:,:,1]
                    ZED_input[0,7,:,:] = data['right'][1][:,:,1]
                    ZED_input[0,8,:,:] = data['left'][0][:,:,2]
                    ZED_input[0,9,:,:] = data['left'][1][:,:,2]
                    ZED_input[0,10,:,:] = data['right'][0][:,:,2]
                    ZED_input[0,11,:,:] = data['right'][1][:,:,2]
                    

                else: # flip left-right
                    ZED_input[0,0,:,:] = data['right_flip'][0][:,:,0]
                    ZED_input[0,1,:,:] = data['right_flip'][1][:,:,0]
                    ZED_input[0,2,:,:] = data['left_flip'][0][:,:,0]
                    ZED_input[0,3,:,:] = data['left_flip'][1][:,:,0]
                    ZED_input[0,4,:,:] = data['right_flip'][0][:,:,1]
                    ZED_input[0,5,:,:] = data['right_flip'][1][:,:,1]
                    ZED_input[0,6,:,:] = data['left_flip'][0][:,:,1]
                    ZED_input[0,7,:,:] = data['left_flip'][1][:,:,1]
                    ZED_input[0,8,:,:] = data['right_flip'][0][:,:,2]
                    ZED_input[0,9,:,:] = data['right_flip'][1][:,:,2]
                    ZED_input[0,10,:,:] = data['left_flip'][0][:,:,2]
                    ZED_input[0,11,:,:] = data['left_flip'][1][:,:,2]

                    for i in range(len(target_data)/2):
                        t = target_data[i]
                        t = t - 49
                        t = -t
                        t = t + 49
                        target_data[i] = t

                #solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] -= 128
                ZED_input[:,:,:,:] /= 255.0
                ZED_input[:,:,:,:] -= 0.5

                if False: #camera_dropout:
                    ri = random.randint(0,5)
                    if ri == 0:
                        #print 'here 0'
                        #time.sleep(0.1)
                        for e in [0,1,4,5,8,9]:
                            ZED_input[0,e,:,:] *= 0
                    elif ri == 1:
                        #print 'here 1'
                        #time.sleep(0.1)
                        for e in [2,3,6,7,10,11]:
                            ZED_input[0,2:3,:,:] *= 0
                    else:
                        pass


                Direct = 0.
                Follow = 0.
                Play = 0.
                Furtive = 0.
                Caf = 0.
                Racing = 0

                if 'follow' in data['path']:
                    Follow = 1.0
                if 'direct' in data['path']:
                    Direct = 1.0
                if 'play' in data['path']:
                    Play = 1.0
                if 'furtive' in data['path']:
                    Furtive = 1.0
                if 'caffe' in data['path']:
                    Caf = 1.0
                if 'racing' in data['path']:
                    Racing = 1.0
                    Direct = 1.0

                meta_input[0,0,:,:] = Racing#target_data[0]/99. #current steer
                meta_input[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
                meta_input[0,2,:,:] = Follow
                meta_input[0,3,:,:] = Direct
                meta_input[0,4,:,:] = Play
                meta_input[0,5,:,:] = Furtive

                for i in range(len(target_data)):
                    steer_motor_target_data[0,i] = target_data[i]/99.
                
                x_train = {'ZED_input': ZED_input, 'meta_input': meta_input}
                y_train = {'steer_motor_target_data': steer_motor_target_data}
                
                result = {'x_train':x_train, 'y_train': y_train}


            if show_data:
                for i in range(len(data['left'])):
                    img = data['left'][i].copy()
                    steer = data['steer'][i]
                    motor = data['motor'][i]
                    gyro_x = data['gyro_x'][i]
                    gyro_yz_mag = data['gyro_yz_mag'][i]

                    apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
                    apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
                    apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
                    apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

                    cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
        return result

    return None

# data_set: generated data set from training_data_generator_{version}
def visualize_data_model_version_1(data_set,flip):
    figure_caption = 'ZED_input'
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    for i in range(2):
        data_img_shape = np.shape(x_train['ZED_input'])
        img = np.zeros((data_img_shape[2],data_img_shape[3],3))

        img[:,:,0] = x_train['ZED_input'][0,i,:,:].copy()
        img[:,:,1] = x_train['ZED_input'][0,i+4,:,:].copy()
        img[:,:,2] = x_train['ZED_input'][0,i+8,:,:].copy()


        img = z2o(img)

        steer = y_train['steer_motor_target_data'][0,i]*99.
        motor = y_train['steer_motor_target_data'][0,i+10]*99.

        if flip:
            steer_rect_color = blue_green
        else:
            steer_rect_color = blue
        #print(img.min(),img.max())
        apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
        apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
        mi(img,layer_to_use,img_title=d2s(x_train['meta_input'][0,:,0,0]))

        pause(0.001)
        #cv2.imshow('left',cv2.cvtColor((255*img).astype(np.uint8),cv2.COLOR_RGB2BGR))#.astype('uint8')
        #if cv2.waitKey(33) & 0xFF == ord('q'):
        #    break
    return True    
    
    
    
# model version 1b
def training_data_generator_1b(data,flip,show_data=False, camera_dropout=False):
    result = {}
    x_train = {}
    y_train = {}
    #return True
    if 'left' in data:
        if len(data['left']) >= 10:

            if type(data['left'][0]) == np.ndarray:
                target_data = data['steer'][:10]
                target_data += data['motor'][:10]
                #mi(data['left'][0][:,:,:],'left')
                #mi(data['right'][0][:,:,:],'right')
                im_h, im_w, _ = data['left'][0].shape
                ZED_input = np.zeros((1, 12, im_h, im_w))
                meta_input = np.zeros((1,6, 14, 26))
                steer_motor_target_data = np.zeros((1,20))                
                
                if not flip:
                    ZED_input[0,0,:,:] = data['left'][0][:,:,0]
                    ZED_input[0,1,:,:] = data['left'][1][:,:,0]
                    ZED_input[0,2,:,:] = data['right'][0][:,:,0]
                    ZED_input[0,3,:,:] = data['right'][1][:,:,0]
                    ZED_input[0,4,:,:] = data['left'][0][:,:,1]
                    ZED_input[0,5,:,:] = data['left'][1][:,:,1]
                    ZED_input[0,6,:,:] = data['right'][0][:,:,1]
                    ZED_input[0,7,:,:] = data['right'][1][:,:,1]
                    ZED_input[0,8,:,:] = data['left'][0][:,:,2]
                    ZED_input[0,9,:,:] = data['left'][1][:,:,2]
                    ZED_input[0,10,:,:] = data['right'][0][:,:,2]
                    ZED_input[0,11,:,:] = data['right'][1][:,:,2]

                else: # flip left-right
                    ZED_input[0,0,:,:] = data['right_flip'][0][:,:,0]
                    ZED_input[0,1,:,:] = data['right_flip'][1][:,:,0]
                    ZED_input[0,2,:,:] = data['left_flip'][0][:,:,0]
                    ZED_input[0,3,:,:] = data['left_flip'][1][:,:,0]
                    ZED_input[0,4,:,:] = data['right_flip'][0][:,:,1]
                    ZED_input[0,5,:,:] = data['right_flip'][1][:,:,1]
                    ZED_input[0,6,:,:] = data['left_flip'][0][:,:,1]
                    ZED_input[0,7,:,:] = data['left_flip'][1][:,:,1]
                    ZED_input[0,8,:,:] = data['right_flip'][0][:,:,2]
                    ZED_input[0,9,:,:] = data['right_flip'][1][:,:,2]
                    ZED_input[0,10,:,:] = data['left_flip'][0][:,:,2]
                    ZED_input[0,11,:,:] = data['left_flip'][1][:,:,2]
                    #solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['right_flip'][0][:,:,0]

                    for i in range(len(target_data)/2):
                        t = target_data[i]
                        t = t - 49
                        t = -t
                        t = t + 49
                        target_data[i] = t
                
                #print type(solver.net.blobs['ZED_data_pool2'].data[0,0,0,0])
                if False:
                    ZED_input[:,:,:,:] /= 255.0
                    ZED_input[:,:,:,:] -= 0.5
                
                
                Direct = 0.
                Follow = 0.
                Play = 0.
                Furtive = 0.
                Caf = 0.
                Racing = 0

                if 'follow' in data['path']:
                    Follow = 1.0
                if 'direct' in data['path']:
                    Direct = 1.0
                if 'play' in data['path']:
                    Play = 1.0
                if 'furtive' in data['path']:
                    Furtive = 1.0
                if 'caffe' in data['path']:
                    Caf = 1.0
                if 'racing' in data['path']:
                    Racing = 1.0
                    Direct = 1.0

                meta_input[0,0,:,:] = Racing#target_data[0]/99. #current steer
                meta_input[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
                meta_input[0,2,:,:] = Follow
                meta_input[0,3,:,:] = Direct
                meta_input[0,4,:,:] = Play
                meta_input[0,5,:,:] = Furtive

                for i in range(len(target_data)):
                    steer_motor_target_data[0,i] = target_data[i]/99.
                
                x_train = {'ZED_input': ZED_input, 'meta_input': meta_input}
                y_train = {'steer_motor_target_data': steer_motor_target_data}
                
                result = {'x_train':x_train, 'y_train': y_train}

            if show_data:
                for i in range(len(data['left'])):
                    img = data['left'][i].copy()
                    steer = data['steer'][i]
                    motor = data['motor'][i]
                    gyro_x = data['gyro_x'][i]
                    gyro_yz_mag = data['gyro_yz_mag'][i]

                    apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
                    apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
                    apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
                    apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

                    cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
        return result

    return None

def visualize_data_model_version_1b(data_set,flip):
    figure_caption = 'ZED_input'
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    r = 2
    for i in range(2):

        data_img_shape = np.shape(x_train['ZED_input'])
        img = np.zeros((data_img_shape[2],2*data_img_shape[3],3))

        img[:,:data_img_shape[3],0] = x_train['ZED_input'][0,r+i,:,:].copy()
        img[:,:data_img_shape[3],1] = x_train['ZED_input'][0,r+i+4,:,:].copy()
        img[:,:data_img_shape[3],2] = x_train['ZED_input'][0,r+i+8,:,:].copy()
        img[:,data_img_shape[3]:,0] = x_train['ZED_input'][0,i,:,:].copy()
        img[:,data_img_shape[3]:,1] = x_train['ZED_input'][0,i+4,:,:].copy()
        img[:,data_img_shape[3]:,2] = x_train['ZED_input'][0,i+8,:,:].copy()

        img = z2o(img)
        steer = y_train['steer_motor_target_data'][0,i]*99.
        motor = y_train['steer_motor_target_data'][0,i+10]*99.

        if flip:
            steer_rect_color = blue_green
        else:
            steer_rect_color = blue
        #apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
        #apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
        figure(figure_caption,figsize=(4,2))
        mi(img,figure_caption,img_title=d2s(x_train['meta_input'][0,:,0,0],flip))
        #print solver.net.blobs['steer_motor_target_data'].data[:]
        pause(0.0001)

    return True

def training_data_generator_2(data,flip,show_data=False,camera_dropout=False):
    result = {}
    x_train = {}
    y_train = {}
    if 'left' in data:
        if len(data['left']) >= 10:

            if type(data['left'][0]) == np.ndarray:
                steer0 = data['steer'][0]
                motor0 = data['motor'][0]
                target_data = list(array(data['steer'][:]) - steer0)
                target_data += list(array(data['motor'][:]) - motor0)
                target_data += list(array(data['gyro_yz_mag'][:])/10.)
                _, _, im_h, im_w = data['left'][0].shape
                ZED_input = np.zeros((1, 18, im_h, im_w))
                meta_input = np.zeros((1,8, 14, 26))
                steer_motor_target_data = np.zeros((1,60)) 

                if not flip:
                    ZED_input[0,0,:,:] = data['left'][0][:,:,0]
                    ZED_input[0,1,:,:] = data['left'][1][:,:,0]
                    ZED_input[0,2,:,:] = data['left'][2][:,:,0]
                    ZED_input[0,3,:,:] = data['right'][0][:,:,0]
                    ZED_input[0,4,:,:] = data['right'][1][:,:,0]
                    ZED_input[0,5,:,:] = data['right'][2][:,:,0]
                    ZED_input[0,6,:,:] = data['left'][0][:,:,1]
                    ZED_input[0,7,:,:] = data['left'][1][:,:,1]
                    ZED_input[0,8,:,:] = data['left'][2][:,:,1]
                    ZED_input[0,9,:,:] = data['right'][0][:,:,1]
                    ZED_input[0,10,:,:] = data['right'][1][:,:,1]
                    ZED_input[0,11,:,:] = data['right'][2][:,:,1]
                    ZED_input[0,12,:,:] = data['left'][0][:,:,2]
                    ZED_input[0,13,:,:] = data['left'][1][:,:,2]
                    ZED_input[0,14,:,:] = data['left'][2][:,:,2]
                    ZED_input[0,15,:,:] = data['right'][0][:,:,2]
                    ZED_input[0,16,:,:] = data['right'][1][:,:,2]
                    ZED_input[0,17,:,:] = data['right'][2][:,:,2]
                else: # flip left-right
                    ZED_input[0,0,:,:] = data['right_flip'][0][:,:,0]
                    ZED_input[0,1,:,:] = data['right_flip'][1][:,:,0]
                    ZED_input[0,2,:,:] = data['right_flip'][2][:,:,0]
                    ZED_input[0,3,:,:] = data['left_flip'][0][:,:,0]
                    ZED_input[0,4,:,:] = data['left_flip'][1][:,:,0]
                    ZED_input[0,5,:,:] = data['left_flip'][2][:,:,0]
                    ZED_input[0,6,:,:] = data['right_flip'][0][:,:,1]
                    ZED_input[0,7,:,:] = data['right_flip'][1][:,:,1]
                    ZED_input[0,8,:,:] = data['right_flip'][2][:,:,1]
                    ZED_input[0,9,:,:] = data['left_flip'][0][:,:,1]
                    ZED_input[0,10,:,:] = data['left_flip'][1][:,:,1]
                    ZED_input[0,11,:,:] = data['left_flip'][2][:,:,1]
                    ZED_input[0,12,:,:] = data['right_flip'][0][:,:,2]
                    ZED_input[0,13,:,:] = data['right_flip'][1][:,:,2]
                    ZED_input[0,14,:,:] = data['right_flip'][2][:,:,2]
                    ZED_input[0,15,:,:] = data['left_flip'][0][:,:,2]
                    ZED_input[0,16,:,:] = data['left_flip'][1][:,:,2]
                    ZED_input[0,17,:,:] = data['left_flip'][2][:,:,2]

                    for i in range(30):
                        t = target_data[i]
                        #t = t - 49
                        t = -t
                        #t = t + 49
                        target_data[i] = t

                ZED_input[:,:,:,:] /= 255.0
                ZED_input[:,:,:,:] -= 0.5

                Direct = 0.
                Follow = 0.
                Play = 0.
                Furtive = 0.
                Caf = 0.
                Racing = 0

                if 'follow' in data['path']:
                    Follow = 1.0
                if 'direct' in data['path']:
                    Direct = 1.0
                if 'play' in data['path']:
                    Play = 1.0
                if 'furtive' in data['path']:
                    Furtive = 1.0
                if 'caffe' in data['path']:
                    Caf = 1.0
                if 'racing' in data['path']:
                    Racing = 1.0
                    Direct = 1.0

                meta_input[0,0,:,:] = Racing#target_data[0]/99. #current steer
                meta_input[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
                meta_input[0,2,:,:] = Follow
                meta_input[0,3,:,:] = Direct
                meta_input[0,4,:,:] = Play
                meta_input[0,5,:,:] = Furtive
                meta_input[0,6,:,:] = steer0/99.
                meta_input[0,7,:,:] = motor0/99.

                for i in range(len(target_data)):
                    steer_motor_target_data[0,i] = target_data[i]/99.

                x_train = {'ZED_input': ZED_input, 'meta_input': meta_input}
                y_train = {'steer_motor_target_data': steer_motor_target_data}
                
                result = {'x_train':x_train, 'y_train': y_train}


            if show_data:
                for i in range(len(data['left'])):
                    img = data['left'][i].copy()
                    steer = data['steer'][i]
                    motor = data['motor'][i]
                    gyro_x = data['gyro_x'][i]
                    gyro_yz_mag = data['gyro_yz_mag'][i]

                    apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
                    apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
                    apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
                    apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

                    cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
                    if cv2.waitKey(33) & 0xFF == ord('q'):
                        break
        return result

    return None

def visualize_data_model_version_2(data_set, flip):        
    
    
    figure_caption = 'ZED_input'
    x_train = data_set['x_train']
    y_train = data_set['y_train']
    
    """
    TODO: load model and plot the prediction output against the training target
    figure(figure_caption+'_')
    clf()
    plot(y_train['steer_motor_target_data'][0,:30],'ro-')
    plot((y_train['steer_motor_target_data'][0,30:60],'rx-')
    plot((y_train['steer_motor_target_data'][0,60:],'r')
    plot(solver.net.blobs['ip2'].data[0,:30],'bo-')
    plot(solver.net.blobs['ip2'].data[0,30:60],'bx-')
    plot(solver.net.blobs['ip2'].data[0,60:],'b')
    """    
    r = 3
    for i in range(3):

        data_img_shape = np.shape(x_train['ZED_input'])
        img = np.zeros((data_img_shape[2],2*data_img_shape[3],3))

        img[:,:data_img_shape[3],0] = x_train['ZED_input'][0,r+i,:,:].copy()
        img[:,:data_img_shape[3],1] = x_train['ZED_input'][0,r+i+6,:,:].copy()
        img[:,:data_img_shape[3],2] = x_train['ZED_input'][0,r+i+12,:,:].copy()
        img[:,data_img_shape[3]:,0] = x_train['ZED_input'][0,i,:,:].copy()
        img[:,data_img_shape[3]:,1] = x_train['ZED_input'][0,i+6,:,:].copy()
        img[:,data_img_shape[3]:,2] = x_train['ZED_input'][0,i+12,:,:].copy()

        img = z2o(img)
        steer = y_train['steer_motor_target_data'][0,i]*99.
        motor = y_train['steer_motor_target_data'][0,i+10]*99.

        if flip:
            steer_rect_color = blue_green
        else:
            steer_rect_color = blue
        #apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
        #apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
        mi(img,layer_to_use,img_title=d2s(x_train['meta_input'][0,:,0,0],flip))


        pause(0.001)

    return True

