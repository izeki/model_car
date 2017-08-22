from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, \
                         MaxPooling2D, Dense, ZeroPadding2D, Flatten, concatenate
from keras import regularizers
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers import Activation, Merge
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
import numpy as np

#### keras does not implement the EuclideanLoss layer...################################
#
#      https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py
#
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)    

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
                                          name=name+'_'+str(i))(splittensor(
                                                                                            axis=1,
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



def get_model(version, channel=3, meta_label=6, input_width=672, input_height=376, phase='train'):
    if version == 'version 1b':
        return get_model_1b(channel, meta_label, input_width, input_height, phase)
    elif version == 'squeeze_net':
        return get_model_squeez_net(channel, meta_label, input_width, input_height, phase)
    else:
        assert(False)

def get_model_1b(channel=3, meta_label=6, input_width=672, input_height=376, phase='train'):
    ZED_data = Input(shape=(channel*4, input_height, input_width), name='ZED_input')
    metadata =  Input(shape=(meta_label, 14, 26), name='meta_input')
    #steer_motor_target_data = Input(shape=(1,20), name='steer_motor_target_data')    
    
    ZED_data_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(ZED_data)
    ZED_data_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='ZED_data_pool1')(ZED_data_pad)
    ZED_data_pool1 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(ZED_data_pool1)
    ZED_data_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='ZED_data_pool2')(ZED_data_pool1)
    #ZED_data_pool2_scale = BatchNormalization(axis=1, name='ZED_data_pool2_scale')(ZED_data_pool2)
    ZED_data_pool2_scale = Scale(axis=1, name='ZED_data_pool2_scale')(ZED_data_pool2)
    
    conv1 = Conv2D(filters=96, kernel_size=11, strides=(3,3), padding='valid', activation='relu', data_format='channels_first', name='conv1')(ZED_data_pool2_scale)
    conv1 = ZeroPadding2D(padding=(1, 0), data_format='channels_first')(conv1)
    conv1_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv1_pool')(conv1)
    conv1_metadata_concat = concatenate([conv1_pool, metadata], axis=-3, name='conv1_metadata_concat')
    
    conv2 = conv2Dgroup(group=2, axis=-3, filters=256, kernel_size=3, strides=(2,2), padding='valid', activation='relu', data_format='channels_first', name='conv2')(conv1_metadata_concat)
    conv2 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(conv2)
    conv2_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv2_pool')(conv2)
    conv2_pool = Flatten()(conv2_pool)
    
    ip1 = Dense(units=512, activation='relu', use_bias=False, name='ip1')(conv2_pool)
    ip2 = Dense(units=20, use_bias=False, name='ip2')(ip1)
    
    #if phase == 'train':
    #    euclidean = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([steer_motor_target_data, ip2])
    
    if phase == 'train':
        model = Model(inputs=[ZED_data, metadata], outputs=ip2) 
        #model = Model(inputs=[ZED_data, metadata, steer_motor_target_data], outputs=euclidean) 
    elif phase == 'test':
        model = Model(inputs=[ZED_data, metadata], outputs=ip2) 
    else:
        model = None
    return model

def load_model_weight(model, weights_path):    
    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(weights_path, encoding='latin1').item()
        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                if (layer.name=='ip1' or layer.name=='ip2'):
                    layer.set_weights((layer_weights['weights'],))
                else:
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
    
    
def fire(squeeze_planes, expand1x1_planes, expand3x3_planes):
    def f(input):
        squeeze1x1 = Conv2D(filters=squeeze_planes, kernel_size=1, strides=(2,2), padding='valid', activation='relu', data_format='channels_first', name='squeeze1x1')(input)
        expand1x1 = Conv2D(filters=expand1x1_planes, kernel_size=1, strides=(2,2), padding='valid', activation='relu', data_format='channels_first', name='squeeze1x1')(squeeze1x1)
        expand3x3 = Conv2D(filters=expand3x3_planes, kernel_size=3, strides=(2,2), padding='same', activation='relu', data_format='channels_first', name='squeeze1x1')(squeeze1x1)
        #expand3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(expand3x3)
        return concatenate([expand1x1, expand3x3], axis=1, name='concat')
    return f    


def get_model_squeez_net(channel=3, meta_label=6, input_width=672, input_height=376,  phase='train'):
    N_STEPS = 10
    ZED_data = Input(shape=(channel*4, input_height, input_width), name='ZED_input')
    metadata = Input(shape=(meta_label, 11, 20), name='meta_input')
    
    conv1 = Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='valid', activation='relu', data_format='channels_first', name='conv1')(ZED_data)
    conv1_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv1_pool')(conv1)

    fire1 = fire(16, 64, 64)(conv1_pool)
    fire2 = fire(16, 64, 64)(fire1)
    fire_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='fire_pool1')(fire2)
    fire_pool1_metadata_concat = concatenate([fire_pool1, metadata], axis=-3, name='fire_pool1_metadata_concat')    
    
    fire3 = fire(32, 128, 128)(fire_pool1_metadata_concat)
    fire4 = fire(32, 128, 128)(fire3)
    fire_pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='fire_pool2')(fire4)
    fire5 = fire(48, 192, 192)(fire_pool2)
    fire6 = fire(48, 192, 192)(fire5)
    fire7 = fire(64, 256, 256)(fire6)
    fire8 = fire(64, 256, 256)(fire7)
    
    drop1 = Dropout(rate=0.5, name='drop1')(fire8)
    conv2 = Conv2D(filters=2 * N_STEPS, kernel_size=1, strides=(2,2), padding='valid', data_format='channels_first', name='conv2')(drop1)
    avg_pool1 = AveragePooling2D(pool_size=(5, 5), strides=(6,6), padding='valid', data_format='channels_first', name='avg_pool1')(conv2)
    reshape1 = Reshape((avg_pool1.shape[0].value, -1))(avg_pool1)
    
    if phase == 'train':
        model = Model(inputs=[ZED_data, metadata], outputs=reshape1) 
    elif phase == 'test':
        model = Model(inputs=[ZED_data, metadata], outputs=reshape1) 
    else:
        model = None
    return model
    