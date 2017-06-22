from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, MaxPooling2D, Dense, ZeroPadding2D,  Reshape, concatenate
from keras import regularizers
from keras.layers.core import Lambda
from keras.layers import Activation, Merge
from keras import backend as K

"""
Karl's model car model:
name: "z2_color_4_layers"
layer {
	name: "steer_motor_target_data"
	type: "DummyData"
	top: "steer_motor_target_data"
	dummy_data_param {
		shape {
			dim: 1
			dim: 20
		}
	}
}

layer {
        name: "metadata"
        type: "DummyData"
        top: "metadata"
        dummy_data_param {
                shape {
                        dim: 1
                        dim: 6
                        dim: 14
                        dim: 26
                }
        }
}

layer {
	name: "ZED_data"
	type: "DummyData"
	top: "ZED_data"
	dummy_data_param {
		shape {
			dim: 1
			dim: 12
			dim: 376
			dim: 672
		}
	}
}

layer {
	name: "ZED_data_pool1"
	type: "Pooling"
	bottom: "ZED_data" #"MVN" #
	top: "ZED_data_pool1"
	pooling_param {
		pool: AVE
		kernel_size: 3
		stride: 2
		pad: 0
	}
}

layer {
	name: "ZED_data_pool2"
	type: "Pooling"
	bottom: "ZED_data_pool1"
	top: "ZED_data_pool2"
	pooling_param {
		pool: AVE
		kernel_size: 3
		stride: 2
		pad: 0
	}
}

layer {
  name: "ZED_data_pool2_scale"
  type: "Scale"
  bottom: "ZED_data_pool2"
  top: "ZED_data_pool2_scale"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  scale_param {
    filler {
      value: 0.003921    }
    bias_term: true
    bias_filler {
      value: -0.5
    }
  }
}
	
###################### Convolutional Layer Set 'conv1' ######################
#
layer {
	name: "conv1"
	type: "Convolution"
	bottom: "ZED_data_pool2_scale"
	top: "conv1"
	convolution_param {
		num_output: 96
		group: 1
		kernel_size: 11
		stride: 3
		pad: 0
		weight_filler {
			type: "gaussian" 
			std: 0.00001
		}
	}
}
	
layer {
	name: "conv1_relu"
	type: "ReLU"
	bottom: "conv1"
	top: "conv1"
}
	
layer {
	name: "conv1_pool"
	type: "Pooling"
	bottom: "conv1"
	top: "conv1_pool"
	pooling_param {
		pool: MAX
		kernel_size: 3
		stride: 2
		pad: 0
	}
}
	
############################################################

layer {
  name: "conv1_metadata_concat"
  type: "Concat"
  bottom: "conv1_pool"
  bottom: "metadata"
  top: "conv1_metadata_concat"
  concat_param {
    axis: 1
  }     
}               
                        


###################### Convolutional Layer Set 'conv2' ######################
#
layer {
	name: "conv2"
	type: "Convolution"
	bottom: "conv1_metadata_concat"
	top: "conv2"
	convolution_param {
		num_output: 256
		group: 2
		kernel_size: 3
		stride: 2
		pad: 0
		weight_filler {
			type: "gaussian" 
			std: 0.1
		}
	}
}
	
layer {
	name: "conv2_relu"
	type: "ReLU"
	bottom: "conv2"
	top: "conv2"
}
	
layer {
	name: "conv2_pool"
	type: "Pooling"
	bottom: "conv2"
	top: "conv2_pool"
	pooling_param {
		pool: MAX
		kernel_size: 3
		stride: 2
		pad: 0
	}
}
	
############################################################


###################### IP Layer Set 'ip1' ######################
#
layer {
	name: "ip1"
	type: "InnerProduct"
	bottom: "conv2_pool"
	top: "ip1"
	inner_product_param {
		num_output: 512
		weight_filler {
			type: "xavier" 
		}
	}
}
	
layer {
	name: "ip1_relu"
	type: "ReLU"
	bottom: "ip1"
	top: "ip1"
}
	
############################################################


###################### IP Layer Set 'ip2' ######################
#
layer {
	name: "ip2"
	type: "InnerProduct"
	bottom: "ip1"
	top: "ip2"
	inner_product_param {
		num_output: 20
		weight_filler {
			type: "xavier" 
		}
	}
}
"""


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
        return concatenate([
                         Conv2D(filters=filters // group, kernel_size=kernel_size, 
                                strides=strides, padding=padding, activation=activation, 
                                data_format=data_format, name=name+'_'+str(i)) (
                             splittensor(axis=1,
                                         ratio_split=group,
                                         id_split=i)(input))
                         for i in range(group)
                         ], axis=axis)

    return f



def get_model(channel=3, meta_label=6, input_width=672, input_height=376, phase='train'):
    ZED_data = Input(shape=(channel*4, input_height, input_width), name='ZED_data')
    metadata =  Input(shape=(meta_label, 14, 26), name='metadata')
    steer_motor_target_data = Input(shape=(1,20), name='steer_motor_target_data')    
    
    ZED_data_pad = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(ZED_data)
    ZED_data_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='ZED_data_pool1')(ZED_data_pad)
    ZED_data_pool1 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(ZED_data_pool1)
    ZED_data_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='ZED_data_pool2')(ZED_data_pool1)
    ZED_data_pool2_scale = BatchNormalization(axis=1, name='ZED_data_pool2_scale')(ZED_data_pool2)
    
    conv1 = Conv2D(filters=96, kernel_size=11, strides=(3,3), padding='valid', activation='relu', data_format='channels_first', name='conv1')(ZED_data_pool2_scale)
    conv1 = ZeroPadding2D(padding=(1, 0), data_format='channels_first')(conv1)
    conv1_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv1_pool')(conv1)
    conv1_metadata_concat = concatenate([conv1_pool, metadata], axis=-3, name='conv1_metadata_concat')
    
    conv2 = conv2Dgroup(group=2, axis=-3, filters=256, kernel_size=3, strides=(2,2), padding='valid', activation='relu', data_format='channels_first', name='conv2')(conv1_metadata_concat)
    conv2 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(conv2)
    conv2_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv2_pool')(conv2)
    conv2_pool = Reshape((256*3*6,1))(conv2_pool)
    
    ip1 = Dense(units=512, activation='relu', use_bias=False, name='ip1')(conv2_pool)
    ip2 = Dense(units=20, use_bias=False, name='ip2')(ip1)
    
    if phase == 'train':
        euclidean = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([steer_motor_target_data, ip2])
    
    if phase == 'train':
        model = Model(inputs=[ZED_data, metadata, steer_motor_target_data], outputs=euclidean) 
    elif phase == 'test':
        model = Model(inputs=[ZED_data, metadata], outputs=ip2) 
    else:
        model = None
    return model
    
    
