from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, concatenate
from keras import regularizers
from keras.layers import Activation
import keras as k


"""
Karl's model car model:
name: "z2_color_4_layers"
input: "ZED_data_pool2"
input_shape {
    dim: 1
    dim: 12
    dim: 94
    dim: 168
}

input: "metadata"
input_shape {
    dim: 1
    dim: 6
    dim: 14
    dim: 26
}

input: "steer_motor_target_data"
input_shape {
    dim: 1
    dim: 20
}

###################### Convolutional Layer Set 'conv1' ######################
#
layer {
	name: "conv1"
	type: "Convolution"
	bottom: "ZED_data_pool2"
	top: "conv1"
	convolution_param {
		num_output: 96
		group: 1
		kernel_size: 11
		stride: 3
		pad_h: 0
		pad_w: 0
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
		pad_h: 0
		pad_w: 0
	}
}
	
#
############################################################



layer {
	type: 'Concat'
	name: 'conv1_metadata_concat'
	bottom: "conv1_pool"
	bottom: "metadata"
	top: 'conv1_metadata_concat'
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
		pad_h: 0
		pad_w: 0
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
		pad_h: 0
		pad_w: 0
	}
}
	
#
############################################################

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

layer {
	name: "euclidean"
	type: "EuclideanLoss"
	bottom: "steer_motor_target_data"
	bottom: "ip2"
	top: "euclidean"
	loss_weight: 1
}

"""


#### keras does not implement the EuclideanLoss layer...################################

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)    


def get_model(channel=3, meta_label=6, input_width=672, input_height=376, phase='train'):
    ZED_data_left_frame1 = Input(shape=(channel,input_width, input_height), name='ZED_data_left_frame1')
    ZED_data_left_frame2 = Input(shape=(channel,input_width, input_height), name='ZED_data_left_frame2')
    ZED_data_right_frame1 = Input(shape=(channel,input_width, input_height), name='ZED_data_right_frame1')
    ZED_data_right_frame2 = Input(shape=(channel,input_width, input_height), name='ZED_data_right_frame2')
    ZED_data = concatenate([ZED_data_left_frame1, ZED_data_left_frame2, ZED_data_right_frame1, ZED_data_right_frame2], axis=-3, name='ZED_data')
    metadata =  Input(shape=(meta_label, input_width, input_height), name='metadata')
    steer_motor_target_data = Input(shape=(1,20), name='steer_motor_target_data')    
   
    conv1 = Conv2D(filters=96, kernel_size=11, strides=(3,3), padding='valid', activation='relu', data_format='channels_first', name='conv1')(ZED_data)
    conv1_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv1_pool')(conv1)
    
    conv1_metadata_concat = concatenate([ZED_data, metadata], axis=-3, name='conv1_metadata_concat')
    conv2 = Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='valid', activation='relu', data_format='channels_first', name='conv2')(conv1_metadata_concat)
    conv2_pool = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', data_format='channels_first', name='conv1_pool')(conv2)
    
    ip1 = Dense(units=512, activation='relu', name='ip1')(conv2_pool)
    ip2 = Dense(units=20, activation='relu', name='ip1')(ip1)
    
    if phase == 'train':
        euclidean = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([steer_motor_target_data, ip2])
    
    if phase == 'train':
        model = Model(inputs=[ZED_data, metadata, steer_motor_target_data], outputs=euclidean) 
    elif phase == 'test':
        model = Model(inputs=[ZED_data, metadata], outputs=ip2) 
    else:
        model = None
    return model
    
    
