"""z2_color implementation with batch normalization."""
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, \
                         MaxPooling2D, Dense, ZeroPadding2D, Flatten, concatenate
from keras import regularizers
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers import Activation, Merge

from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from Net import Net
import numpy as np


### keras doesnot implement group convolution ######################################
#
#    https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
#

def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = X.shape[axis].value // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)


def conv2Dgroup(filters, kernel_size, strides, padding, activation, name, data_format, group=1, axis=1, **kwargs):
    def f(input):
        return concatenate(
                        [Conv2D(filters=filters // group, 
                                          kernel_size=kernel_size, 
                                          strides=strides, 
                                          padding=padding, 
                                          activation=activation, 
                                          data_format=data_format, 
                                          name=name+'_'+str(i))(splittensor(axis=1,
                                                                            ratio_split=group,
                                                                            id_split=i)(input))
                         for i in range(group)],
                        axis=axis)

    return f


### keras doesnot implement Scale layer######################################################
#
#  https://flyyufelix.github.io/2017/03/23/caffe-to-keras.html
#

class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    '''
    def __init__(self,  
                        axis=-1, 
                        momentum = 0.9, 
                        beta_initializer ='zero', 
                        gamma_initializer='one', 
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        beta_constraint=None,
                        gamma_constraint=None,
                        **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                            str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                                    axes={self.axis: dim})
        
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                                        name='gamma',
                                                        initializer=self.gamma_initializer,
                                                        regularizer=self.gamma_regularizer,
                                                        constraint=self.gamma_constraint)

        self.beta = self.add_weight(shape=shape,
                                                        name='beta',
                                                        initializer=self.beta_initializer,
                                                        regularizer=self.beta_regularizer,
                                                        constraint=self.beta_constraint)

    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * inputs + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class Z2ColorBatchNorm(Net):
    def __init__(self, meta_label=6, input_width=672, input_height=376):
        super(Z2ColorBatchNorm, self).__init__(meta_label, input_width, input_height)
        self.metadata_size = (14, 26)
    
    def _get_model(self):
        ZED_data = Input(shape=(self.N_CHANNEL*4, self.input_height, self.input_width), name='ZED_input')
        metadata =  Input(shape=(self.meta_label, 14, 26), name='meta_input')
        
        ZED_data_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(ZED_data)        
        ZED_data_pool1 = AveragePooling2D(pool_size=(3, 3), 
                                          strides=(2,2), 
                                          padding='valid', 
                                          data_format='channels_first', 
                                          name='ZED_data_pool1')(ZED_data_pad)        
        ZED_data_pool1 = ZeroPadding2D(padding=(1, 1), 
                                       data_format='channels_first')(ZED_data_pool1)        
        ZED_data_pool2 = AveragePooling2D(pool_size=(3, 3), 
                                          strides=(2,2), 
                                          padding='valid', 
                                          data_format='channels_first', 
                                          name='ZED_data_pool2')(ZED_data_pool1)    
                        
        #ZED_data_pool2_scale = BatchNormalization(axis=1, name='ZED_data_pool2_scale')(ZED_data_pool2)
        ZED_data_pool2_scale = Scale(axis=1, name='ZED_data_pool2_scale')(ZED_data_pool2)
        conv1 = Conv2D(filters=96, 
                       kernel_size=11, 
                       strides=(3,3), 
                       padding='valid', 
                       activation='relu', 
                       data_format='channels_first', 
                       name='conv1')(ZED_data_pool2_scale)
        conv1 = ZeroPadding2D(padding=(1, 0), 
                              data_format='channels_first')(conv1)
        conv1_pool = MaxPooling2D(pool_size=(3, 3), 
                                  strides=(2,2), 
                                  padding='valid', 
                                  data_format='channels_first', 
                                  name='conv1_pool')(conv1)
        conv1_metadata_concat = concatenate([conv1_pool, metadata], axis=-3, name='conv1_metadata_concat')
        conv2 = conv2Dgroup(group=2, 
                            axis=-3, 
                            filters=256, 
                            kernel_size=3, 
                            strides=(2,2), 
                            padding='valid', 
                            activation='relu', 
                            data_format='channels_first', 
                            name='conv2')(conv1_metadata_concat)
        conv2 = ZeroPadding2D(padding=(1, 1), 
                              data_format='channels_first')(conv2)
        conv2_pool = MaxPooling2D(pool_size=(3, 3), 
                                  strides=(2,2), 
                                  padding='valid',
                                  data_format='channels_first',
                                  name='conv2_pool')(conv2)
        conv2_pool = Flatten()(conv2_pool)

        ip1 = Dense(units=512, activation='relu', use_bias=False, name='ip1')(conv2_pool)
        ip2 = Dense(units=20, use_bias=False, name='out')(ip1)
        
        model = Model(inputs=[ZED_data, metadata], outputs=ip2) 
        
        return model
    
    def get_layer_output(self, model_input, training_flag = True):
        get_outputs = K.function([self.net.layers[0].input, 
                                  self.net.layers[9].input, K.learning_phase()],
                                 [self.net.layers[20].output])
        layer_outputs = get_outputs([model_input[0], model_input[1], training_flag])[0]
        return layer_outputs      


def unit_test():
    test_net = Z2ColorBatchNorm(6, 672, 376)
    test_net.model_init()
    test_net.net.summary()
    a = test_net.forward({'ZED_input': np.random.rand(1, 12, 376, 672),
                          'meta_input': np.random.rand(1, 6, 14, 26)})
    
    print(a)


unit_test()
