from keras_model.libs.utils2 import *

import numpy as np

#from nets.SqueezNet import SqueezeNet
from nets.Z2ColorBatchNorm import Z2ColorBatchNorm
import runtime_parameters as rp


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


nframes = None
def init_model():
    global solver, nframes
    print("Loaded "+rp.weights_file_path)
    solver = Z2ColorBatchNorm()
    solver.model_init(rp.weights_file_path)    
    nframes = solver.N_FRAMES
    

init_model()




@static_vars(ai_motor_previous=49, ai_steer_previous=49)
def run_model(camera_input, metadata):
    """
    Runs neural net to get motor and steer data. Scales output to 0 to 100 and applies an IIR filter to smooth the
    performance.

    :param camera_input: Formatted input data from ZED depth camera
    :param metadata: Formatted metadata from user input
    :return: Motor and Steering values
    """
    # Run the neural net
    output = solver.forward({'ZED_input': camera_input,
                             'meta_input': metadata}) 
    if rp.verbose:
        print(output)

    #ai_motor = 100 * output[0][19].data[0]
    #ai_steer = 100 * output[0][9].data[0]
    ai_motor = output[1][0]
    ai_steer = output[0][0] 

    if True: #rp.verbose:
        print('AI Prescale Motor: ' + str(ai_motor))
        print('AI Prescale Steer: ' + str(ai_steer))
    
    ai_motor = int((ai_motor - 49.) * rp.motor_gain + 49.)
    ai_steer = int((ai_steer - 49.) * rp.steer_gain + 49.)

    ai_motor = max(0, ai_motor)
    ai_steer = max(0, ai_steer)
    ai_motor = min(99, ai_motor)
    ai_steer = min(99, ai_steer)


    ai_motor = int((ai_motor + run_model.ai_motor_previous) / 2.0)
    
    if ai_motor > 60:
        ai_motor = 60

    run_model.ai_motor_previous = ai_motor
    

    ai_steer = int((ai_steer + run_model.ai_steer_previous) / 2.0)
    run_model.ai_steer_previous = ai_steer

    return ai_steer, ai_motor




def format_camera_data(left_list, right_list):
    """
    Formats camera data from raw inputs from camera.

    :param l0: left camera data from time step 0
    :param l1: right camera data from time step 1
    :param r0: right camera dataa from time step 0
    :param r1: right camera data from time step 0
    :return: formatted camera data ready for input into pytorch z2color
    """
    camera_start = time.clock()
    camera_data = np.zeros((1,3*2*nframes,376,672))
    listoftensors = []    
    for side in (left_list, right_list):        
        for i in range(nframes):
            listoftensors.append(side[i-2])
    camera_input = np.concatenate(listoftensors, axis=2)
    camera_input = camera_input/255. - 0.5
    camera_input = np.transpose(camera_input, (2,0,1))
    camera_data[0, :, :, :] = camera_input

    return camera_data


def format_metadata(meta_data_label):
    """
    Formats meta data from raw inputs from camera.
    :return:
    """
    metadata = np.zeros((1, 
                         solver.meta_label,
                         solver.metadata_size[0], 
                         solver.metadata_size[1]))
    
    i_label=0

    for label in meta_data_label:
        metadata[0,i_label,:,:]= meta_data_label[label]
        i_label+=1
    return metadata

