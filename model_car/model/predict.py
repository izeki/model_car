import numpy as np
from model import get_model, load_model_weight
from keras.layers import Input

def get_trained_model(version, weights_file_path, input_width=672, input_height=376):
    if version == 'version 1b':
        return get_trained_model_1b(weights_file_path, input_width, input_height)
    else:
        assert(False)

def get_trained_model_1b(weights_path, input_width=672, input_height=376):
    # Returns a model with loaded weights.
    
    model = get_model('version 1b', input_width=input_width,  input_height=input_height, phase='test')
    
    model = load_model_weight(model, weights_path)

    return model


def forward_pass(version, trained_model, ZED_data, meta_data_label, input_width=672, input_height=376):
    if version == 'version 1b':
        return forward_pass_1b(trained_model, ZED_data, meta_data_label, input_width, input_height)
    else:
        return None


def forward_pass_1b(trained_model, ZED_data,  meta_data_label, input_width=672, input_height=376):
    # Runs a forward pass to predict the out of servo and motor
    # Params:
    # trained_model: trained model from get_trained_model
    # ZED_data: ZED camera data, ZED_data =  {'ZED_data_left_frame1':ZED_data_left_frame1, 'ZED_data_left_frame2':ZED_data_left_frame2, 
    #                                                                        'ZED_data_right_frame1':ZED_data_right_frame1, 'ZED_data_right_frame2': ZED_data_right_frame2}
    # meta_data_label: the meta data label: meta_data_label = {'Direct'=Direct, 'Follow'=Follow, 'Play'=Play, 'Furtive'=Furtive, 'Caf'=Cafe, 'Racing'=Racing}
    
    # 2 frames each side
    ZED_input = np.zeros((1, 12, input_height, input_width))
    ZED_input[0,0,:,:]= ZED_data['ZED_data_left_frame1'][:,:,0]
    ZED_input[0,1,:,:]= ZED_data['ZED_data_left_frame2'][:,:,0]
    ZED_input[0,2,:,:]= ZED_data['ZED_data_right_frame1'][:,:,0]
    ZED_input[0,3,:,:]= ZED_data['ZED_data_right_frame2'][:,:,0]
    ZED_input[0,4,:,:]= ZED_data['ZED_data_left_frame1'][:,:,1]
    ZED_input[0,5,:,:]= ZED_data['ZED_data_left_frame2'][:,:,1]
    ZED_input[0,6,:,:]= ZED_data['ZED_data_right_frame1'][:,:,1]
    ZED_input[0,7,:,:]= ZED_data['ZED_data_right_frame2'][:,:,1]
    ZED_input[0,8,:,:]= ZED_data['ZED_data_left_frame1'][:,:,2]
    ZED_input[0,9,:,:]= ZED_data['ZED_data_left_frame2'][:,:,2]
    ZED_input[0,10,:,:]= ZED_data['ZED_data_right_frame1'][:,:,2]
    ZED_input[0,11,:,:]= ZED_data['ZED_data_right_frame2'][:,:,2]
    
    
    meta_input = np.zeros((1,6, 14, 26))
    meta_input[0,0,:,:]= meta_data_label['Racing']
    meta_input[0,1,:,:]= meta_data_label['Caf']
    meta_input[0,2,:,:]= meta_data_label['Follow']
    meta_input[0,3,:,:]= meta_data_label['Direct']
    meta_input[0,4,:,:]= meta_data_label['Play']
    meta_input[0,5,:,:]= meta_data_label['Furtive']
    
    # single sample prediction
    prediction = trained_model.predict_on_batch({'ZED_data':ZED_input, 'metadata':meta_input})
    
    caf_steer = 100*prediction[0,9]
    caf_motor = 100*prediction[0,19]
    
    
    return [caf_steer, caf_motor]
