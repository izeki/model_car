"""z2_color implementation with batch normalization."""
from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, \
                         MaxPooling2D, Dense, ZeroPadding2D, Flatten, concatenate
from keras import optimizers
from keras import regularizers
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers import Activation, Merge
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from Net import Net
import numpy as np

def fire(name, squeeze_planes, expand1x1_planes, expand3x3_planes, **kwargs):
    def f(input):
        squeeze1x1 = Conv2D(filters=squeeze_planes,
                            kernel_size=1, 
                            padding='valid', 
                            activation='relu', 
                            data_format='channels_first', 
                            name='squeeze1x1'+name)(input)
        expand1x1 = Conv2D(filters=expand1x1_planes,
                           kernel_size=1, 
                           padding='valid', 
                           activation='relu', 
                           data_format='channels_first', 
                           name='expand1x1'+name)(squeeze1x1)
        expand3x3 = Conv2D(filters=expand3x3_planes, 
                           kernel_size=3, 
                           padding='valid', 
                           activation='relu', 
                           data_format='channels_first', 
                           name='expand3x3'+name)(squeeze1x1)
        expand3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(expand3x3)
        return concatenate([expand1x1, expand3x3], axis=1, name='concat'+name)
    return f    

    
class SqueezeNet(Net):
    def __init__(self, meta_label=6, input_width=672, input_height=376):
        super(SqueezeNet, self).__init__(meta_label, input_width, input_height)
        self.metadata_size = (11, 20)
    
    def _get_model(self):
        ZED_data = Input(shape=(self.N_CHANNEL*4, self.input_height, self.input_width), name='ZED_input')
        metadata = Input(shape=(self.meta_label, 11, 20), name='meta_input')
        ZED_data_pool1 = AveragePooling2D(pool_size=(2, 2),
                                          strides=(2,2),
                                          padding='valid',
                                          data_format='channels_first',
                                          name='ZED_data_pool1')(ZED_data)
        ZED_data_pool2 = AveragePooling2D(pool_size=(2, 2),
                                          strides=(2,2),
                                          padding='valid',
                                          data_format='channels_first',
                                          name='ZED_data_pool2')(ZED_data_pool1)
        conv1 = Conv2D(filters=64,
                       kernel_size=2,
                       strides=(2,2),
                       padding='valid',
                       activation='relu',
                       data_format='channels_first',
                       name='conv1')(ZED_data_pool2)
        conv1_pool = MaxPooling2D(pool_size=(2, 2),
                                  strides=(2,2),
                                  padding='valid', 
                                  data_format='channels_first',
                                  name='conv1_pool')(conv1)

        fire1 = fire('1', 16, 64, 64)(conv1_pool)
        fire2 = fire('2', 16, 64, 64)(fire1)
        fire_pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2,2),
                                  padding='valid',
                                  data_format='channels_first',
                                  name='fire_pool1')(fire2)
        fire_pool1_metadata_concat = concatenate([fire_pool1, metadata], axis=1, name='fire_pool1_metadata_concat')    

        fire3 = fire('3',32, 128, 128)(fire_pool1_metadata_concat)
        fire4 = fire('4',32, 128, 128)(fire3)
        fire_pool2 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2,2),
                                  padding='valid',
                                  data_format='channels_first',
                                  name='fire_pool2')(fire4)
        fire5 = fire('5',48, 192, 192)(fire_pool2)
        fire6 = fire('6',48, 192, 192)(fire5)
        fire7 = fire('7',64, 256, 256)(fire6)
        fire8 = fire('8',64, 256, 256)(fire7)

        drop1 = Dropout(rate=0.5, name='drop1')(fire8)
        conv2 = Conv2D(filters=2 * self.N_STEPS,
                       kernel_size=1, 
                       padding='valid', 
                       data_format='channels_first',
                       name='conv2')(drop1)
        avg_pool1 = AveragePooling2D(pool_size=(5, 5),
                                     strides=(6,6),
                                     padding='valid', 
                                     data_format='channels_first',
                                     name='avg_pool1')(conv2)

        out = Flatten(name='out')(avg_pool1)


        model = Model(inputs=[ZED_data, metadata], outputs=out) 
        
        return model
    
    def get_layer_output(self, model_input, training_flag = True):
        get_outputs = K.function([self.net.layers[0].input, 
                                  self.net.layers[16].input, K.learning_phase()],
                                 [self.net.layers[52].output])
        layer_outputs = get_outputs([model_input['ZED_input'], model_input['meta_input'], training_flag])[0]
        return layer_outputs 


def unit_test():
    test_net = SqueezeNet(6, 672, 376)
    test_net.model_init()
    test_net.net.summary()
    a = test_net.forward({'ZED_input': np.random.rand(1, 12, 376, 672),
                          'meta_input': np.random.rand(1, 6, 11, 20)})
    
    print(a)


unit_test()
