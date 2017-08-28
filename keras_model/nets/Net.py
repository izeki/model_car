# Basic abstract Net class.
from keras import backend as K
from keras import optimizers

import numpy as np

class Net(object):
    def __init__(self, meta_label=6, input_width=672, input_height=376):
        self.net = None
        self.meta_label = meta_label
        self.input_width = input_width
        self.input_height = input_height
        self.N_CHANNEL = 3
        self.N_FRAMES = 2
        self.N_STEPS = 10
    
    def model_init(self, weight_file_path=None):
        def load_model_weight(model, weight_file_path):    
            def load_tf_weights():
                """ Load pretrained weights converted from Caffe to TF. """

                # 'latin1' enables loading .npy files created with python2
                weights_data = np.load(weight_file_path, encoding='latin1').item()
                for layer in model.layers:
                    if layer.name in weights_data.keys():
                        layer_weights = weights_data[layer.name]
                        if (layer.name=='ip1' or layer.name=='ip2'):
                            layer.set_weights((layer_weights['weights'],))
                        else:
                            layer.set_weights((layer_weights['weights'],  layer_weights['biases']))
            def load_keras_weights():
                """ Load a Keras checkpoint. """
                model.load_weights(weight_file_path)

            if weight_file_path.endswith('.npy'):
                load_tf_weights()
            elif weight_file_path.endswith('.hdf5'):
                load_keras_weights()
            else:
                raise Exception("Unknown weights format.")
            return model
        
        model = self._get_model()
        
        if weight_file_path != None:
            model = load_model_weight(model, weight_file_path)
        
        model.compile(
                loss = 'mean_squared_error', 
                optimizer = optimizers.SGD(
                                    lr = 0.01,
                                    momentum = 0.001, 
                                    decay = 0.000001,
                                    nesterov = True),
                metrics=['accuracy'])
        self.net = model
        
    def _get_model(self):
        raise NotImplementedError
        
    def get_layer_output(self, model_input, training_flag = True):
        raise NotImplementedError
        
    def forward_backward(self, model_input, target_output):
        [loss, accuracy] = self.net.train_on_batch({'ZED_input':model_input['ZED_input'],
                                                    'meta_input':model_input['meta_input']},
                                                   {'out': target_output['steer_motor_target_data']})
        return loss
        
    def forward(self, model_input):
        prediction = self.net.predict_on_batch({'ZED_input':model_input['ZED_input'], 
                                                'meta_input':model_input['meta_input']})
        ai_steer = 100*(prediction[:,9])
        ai_motor = 100*(prediction[:,19])
        return ai_steer.tolist(), ai_motor.tolist()
    
    def save(model_path):
        self.net.save(model_path)