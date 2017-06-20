"""
Create prototxt strings
"""

import caffe
from model_car.utils import *

def conv(top,bottom,num_output,group,kernel_size,stride,pad,weight_filler_type,std=0):
    if type(pad) == int:
        pad_h = pad
        pad_w = pad
    else:
        pad_h = pad[0]
        pad_w = pad[1]
    p = """
layer {
\tname: "TOP"
\ttype: "Convolution"
\tbottom: "BOTTOM"
\ttop: "TOP"
\tconvolution_param {
\t\tnum_output: NUM_OUTPUT
\t\tgroup: NUM_GROUP
\t\tkernel_size: KERNEL_SIZE
\t\tstride: STRIDE
\t\tpad_h: PAD_H
\t\tpad_w: PAD_W
\t\tweight_filler {
\t\t\ttype: "WEIGHT_FILLER_TYPE" STD
\t\t}
\t}
}
    """
    p = p.replace("TOP",top)
    p = p.replace("BOTTOM",bottom)
    p = p.replace("NUM_OUTPUT",str(num_output))
    p = p.replace("NUM_GROUP",str(group))
    p = p.replace("KERNEL_SIZE",str(kernel_size))
    p = p.replace("STRIDE",str(stride))
    p = p.replace("PAD_H",str(pad_h))
    p = p.replace("PAD_W",str(pad_w))
    p = p.replace("WEIGHT_FILLER_TYPE",weight_filler_type)
    if weight_filler_type == 'gaussian':
        p = p.replace("STD","\n\t\t\tstd: "+str(std))
    else:
        p = p.replace("STD","")
    return p

def relu(bottom):
    p = """
layer {
\tname: "BOTTOM_relu"
\ttype: "ReLU"
\tbottom: "BOTTOM"
\ttop: "BOTTOM"
}
    """
    p = p.replace("BOTTOM",bottom)
    return p

def drop(bottom,ratio):
    p = """
layer {
\tname: "BOTTOM_drop"
\ttype: "Dropout"
\tbottom: "BOTTOM"
\ttop: "BOTTOM"
\tdropout_param {
\t\tdropout_ratio: RATIO
\t}
}
    """
    p = p.replace("BOTTOM",bottom)
    p = p.replace("RATIO",str(ratio))
    return p

def deconv(top,bottom,num_output,group,kernel_size,stride,pad,weight_filler_type,std=0):
    return conv(top,bottom,num_output,group,kernel_size,stride,pad,weight_filler_type,std).replace('Convolution','Deconvolution')

def pool(bottom,p_type,kernel_size,stride,pad=0):
    if type(pad) == int:
        pad_h = pad
        pad_w = pad
    else:
        pad_h = pad[0]
        pad_w = pad[1]
    p = """
layer {
\tname: "BOTTOM_pool"
\ttype: "Pooling"
\tbottom: "BOTTOM"
\ttop: "BOTTOM_pool"
\tpooling_param {
\t\tpool: POOL_TYPE
\t\tkernel_size: KERNEL_SIZE
\t\tstride: STRIDE
\t\tpad_h: PAD_H
\t\tpad_w: PAD_W
\t}
}
    """
    p = p.replace("BOTTOM",bottom)
    p = p.replace("POOL_TYPE",p_type)
    p = p.replace("KERNEL_SIZE",str(kernel_size))
    p = p.replace("STRIDE",str(stride))
    p = p.replace("PAD_H",str(pad_h))
    p = p.replace("PAD_W",str(pad_w))
    return p

def conv_layer_set(
    c_top,
    c_bottom,
    c_num_output,
    c_group,
    c_kernel_size,
    c_stride,
    c_pad,
    p_type,
    p_kernel_size,
    p_stride,
    p_pad,
    weight_filler_type,std=0):
    p = """\n###################### Convolutional Layer Set '"""+c_top+"""' ######################\n#"""
    p = p + conv(c_top,c_bottom,c_num_output,c_group,c_kernel_size,c_stride,c_pad,weight_filler_type,std)
    p = p + relu(c_top)
    p = p + pool(c_top,p_type,p_kernel_size,p_stride,p_pad)
    p = p + '\n#\n############################################################\n\n'
    return p

def dummy(top,dims):
    p = """
layer {
\tname: "TOP"
\ttype: "DummyData"
\ttop: "TOP"
\tdummy_data_param {
\t\tshape {
DIMS
\t\t}
\t}
}
    """
    p = p.replace('TOP',top)
    d = ""
    for i in range(len(dims)):
        d = d + d2s('\t\t\tdim:',dims[i])
        if i < len(dims)-1:
            d = d + '\n'
    p = p.replace('DIMS',d)
    return p

def python(top,bottom,module,layer,phase=False):
    p = """
layer {
\ttype: 'Python'
\tname: 'TOP'
\tbottom: 'BOTTOM'
\ttop: 'TOP'
\tpython_param {
\t\tmodule: 'MODULE'
\t\tlayer: 'LAYER'
\t}
        """
    if phase:
        p = p + """
\tinclude {
\t\tphase: PHASE
\t}
"""
    p = p + "\n}"
    p = p.replace('TOP',top)
    p = p.replace('BOTTOM',bottom)
    p = p.replace('MODULE',module)
    p = p.replace('LAYER',layer)
    if phase:
        p = p.replace('PHASE',phase)
    return p

def concat(top,bottom_list,axis):
    p = """
layer {
\ttype: 'Concat'
\tname: 'TOP'
BOTTOM_LIST
\ttop: 'TOP'
\tconcat_param {
\t\taxis: AXIS
\t}
}
        """
    bottom_list_str = ""
    for i in range(len(bottom_list)):
        b = bottom_list[i]
        bottom_list_str += """\tbottom: \""""+b+"""\""""
        if i < len(bottom_list) - 1:
            bottom_list_str += '\n'
    p = p.replace('TOP',top)
    p = p.replace('BOTTOM_LIST',bottom_list_str)
    p = p.replace('AXIS',str(axis))

    return p
def ip(top,bottom,num_output,weight_filler_type,std=0):
    p = """
layer {
\tname: "TOP"
\ttype: "InnerProduct"
\tbottom: "BOTTOM"
\ttop: "TOP"
\tinner_product_param {
\t\tnum_output: NUM_OUTPUT
\t\tweight_filler {
\t\t\ttype: "WEIGHT_FILLER_TYPE" STD
\t\t}
\t}
}
    """
    p = p.replace("TOP",top)
    p = p.replace("BOTTOM",bottom)
    p = p.replace("NUM_OUTPUT",str(num_output))
    p = p.replace("WEIGHT_FILLER_TYPE",weight_filler_type)
    if weight_filler_type == 'gaussian':
        p = p.replace("STD","\n\t\t\tstd "+str(std))
    else:
        p = p.replace("STD","")
    return p

def ip_layer_set(top,bottom,num_output,weight_filler_type,std=0):
    p = """\n###################### IP Layer Set '"""+top+"""' ######################\n#"""
    p = p + ip(top,bottom,num_output,weight_filler_type,std)
    p = p + relu(top)
    p = p + '\n#\n############################################################\n\n'
    return p

def euclidean(top,bottom1,bottom2):
    p = """
layer {
\tname: "TOP"
\ttype: "EuclideanLoss"
\tbottom: "BOTTOM1"
\tbottom: "BOTTOM2"
\ttop: "TOP"
\tloss_weight: 1
}
    """
    p = p.replace("TOP",top)
    p = p.replace("BOTTOM1",bottom1)
    p = p.replace("BOTTOM2",bottom2)
    return p

def solver_proto(model_name):
    p = """
net: "kzpy3/caf2/models/MODEL_NAME/tmp/train_val.prototxt"
test_iter: 1
test_interval: 1000000
test_initialization: false
base_lr: 0.0001  # 0.00005
momentum: 0.01
weight_decay: 0.005
lr_policy: "inv"
gamma: 0.0001
power: 0.75
display: 10000
max_iter: 1000000
snapshot: 10000
snapshot_prefix: "scratch/caf2_models/MODEL_NAME/MODEL_NAME"
    """
    p = p.replace('MODEL_NAME',model_name)
    return p



