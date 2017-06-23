import numpy as np
from model import get_model
from keras.layers import Input

def get_trained_model(weights_path, input_width=672, input_height=376):
    # Returns a model with loaded weights.
    
    model = get_model(input_width=input_width,  input_height=input_height, phase='test')
    
    dir(model)
    
    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(weights_path, encoding='latin1').item()
        for layer in model.layers:
            if layer.name in weights_data.keys():
                print(layer.name) 
                layer_weights = weights_data[layer.name]
                print(layer_weights['weights'].shape)
                layer.set_weights((layer_weights['weights'],  layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(weights_path)

    if weights_path.endswith('.npy'):
        load_tf_weights()
    elif weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    return model

def forward_pass(trained_model, ZED_data, input_width, input_height,  meta_data_label):
    # Runs a forward pass to predict the out of servo and motor
    # Params:
    # trained_model: trained model from get_trained_model
    # ZED_data: ZED camera data, ZED_data =  {'ZED_data_left_frame1':ZED_data_left_frame1, 'ZED_data_left_frame2':ZED_data_left_frame2, 
    #                                                                        'ZED_data_right_frame1':ZED_data_right_frame1, 'ZED_data_right_frame2': ZED_data_right_frame2}
    # meta_data_label: the meta data label: meta_data_label = {'Direct'=Direct, 'Follow'=Follow, 'Play'=Play, 'Furtive'=Furtive, 'Caf'=Cafe, 'Racing'=Racing}
    
    # 2 frames each side
    ZED_input = np.zeros(1, 12, input_width, input_height)
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
    
    
    meta_input = np.zeros(1,6, input_width, input_height)
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
