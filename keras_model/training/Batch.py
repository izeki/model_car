"""Processes data into batches for training and validation."""
import sys

import numpy as np
import matplotlib.pyplot as plt

from Parameters import ARGS
from libs.utils2 import z2o
from libs.vis2 import mi


class Batch:

    def clear(self):
        ''' Clears batch variables before forward pass '''
        self.camera_data = None
        self.metadata = None
        self.target_data = None
        self.names = []
        self.outputs = None
        self.loss = None

    def __init__(self, net):
        self.net = net
        self.camera_data = None
        self.metadata = None
        self.target_data = None
        self.names = None
        self.outputs = None
        self.loss = None
        self.data_ids = None

    def fill(self, data, data_index):
        self.clear()
        self.data_ids = []
        self.camera_data = np.zeros((ARGS.batch_size, 
                                     ARGS.nframes * 6, 
                                     self.net.input_height, 
                                     self.net.input_width))
        self.metadata = np.zeros((ARGS.batch_size, 
                                  self.net.meta_label, 
                                  self.net.metadata_size[0], 
                                  self.net.metadata_size[1]))
        
        self.target_data = np.zeros((ARGS.batch_size, 20))
        
        for data_number in range(ARGS.batch_size):
            data_point = None
            while data_point is None:
                e = data.next(data_index)
                run_code = e[3]
                seg_num = e[0]
                offset = e[1]
                data_point = data.get_data(run_code, seg_num, offset)

            self.data_ids.append((run_code, seg_num, offset))
            self.data_into_batch(data_point, data_number)

    def data_into_batch(self, data, data_number):
        self.names.insert(0, data['name'])
        list_camera_input = []
        # Camera Data        
        for camera in ('left', 'right'):
            for t in range(ARGS.nframes):            
                list_camera_input.append(data[camera][t])
        camera_data = np.concatenate(list_camera_input, axis=2)
        camera_data = camera_data / 255. - 0.5
        camera_data = np.transpose(camera_data, (2,0,1))
        self.camera_data[data_number, :, :, :] = camera_data

        # Behavioral Modes/Metadata
        metadata = np.zeros((self.net.meta_label,
                             self.net.metadata_size[0], 
                             self.net.metadata_size[1]))
        
        metadata_count = 5
        for cur_label in ['furtive', 
                          'play', 
                          'direct', 
                          'follow', 
                          'AI', 
                          'racing']:
            if cur_label == 'AI':
                if data['states'][0]:
                    metadata[metadata_count, :, :] = 1.
                else:
                    metadata[metadata_count, :, :] = 0.
            else:
                if data['labels'][cur_label]:
                    metadata[metadata_count, :, :] = 1.
                else:
                    metadata[metadata_count, :, :] = 0.
            metadata_count -= 1
        self.metadata[data_number, :, :, :] = metadata

        # Figure out which timesteps of labels to get
        s = data['steer']
        m = data['motor']
        r = range(ARGS.stride * ARGS.nsteps - 1, -1, -ARGS.stride)[::-1]
        steer = np.array(s)[r]
        motor = np.array(m)[r]

        # Convert labels to PyTorch Ready Tensors
        steer = steer / 99.
        motor = motor / 99.
        target_data = np.zeros((steer.shape[0]+motor.shape[0]))
        
        target_data[0:len(steer)] = steer
        target_data[len(steer):len(steer) + len(motor)] = motor
        self.target_data[data_number, :] = target_data
    
    def forward_backward(self, data_moment_loss_record, mode='train'):
        if mode == 'train':
            self.loss = self.net.forward_backward({'ZED_input': self.camera_data,
                                                   'meta_input': self.metadata},
                                                  {'steer_motor_target_data': self.target_data})
            self.outputs = self.net.get_layer_output({'ZED_input': self.camera_data,
                                                      'meta_input': self.metadata})
        elif mode == 'eval':
            self.outputs = self.net.get_layer_output({'ZED_input': self.camera_data,
                                                      'meta_input': self.metadata})
            a = self.target_data - self.outputs
            self.loss = np.sqrt(a * a).mean()
        else:
            raise Exception("Unknown mode.")
        
        for b in range(ARGS.batch_size):
            data_id = self.data_ids[b]
            t = self.target_data[b]
            o = self.outputs[b]
            a = self.target_data[b] - self.outputs[b]
            loss = np.sqrt(a * a).mean()
            data_moment_loss_record[(data_id, tuple(t), tuple(o))] = loss        
        

    def display(self):
        if ARGS.display:
            o = self.outputs[0]
            t = self.target_data[0]

            print(
                'Loss:',
                np.round(self.loss,
                         decimals=5))
            a = self.camera_data[0][:]
            b = a.transpose(1, 2, 0)
            h = np.shape(a)[1]
            w = np.shape(a)[2]
            c = np.zeros((10 + h * 2, 10 + 2 * w, 3))
            c[:h, :w, :] = z2o(b[:, :, 3:6])
            c[:h, -w:, :] = z2o(b[:, :, :3])
            c[-h:, :w, :] = z2o(b[:, :, 9:12])
            c[-h:, -w:, :] = z2o(b[:, :, 6:9])
            mi(c, 'cameras')
            print(a.min(), a.max())
            plt.figure('steer')
            plt.clf()
            plt.ylim(-0.05, 1.05)
            plt.xlim(0, len(t))
            plt.plot([-1, 60], [0.49, 0.49], 'k')  # plot in black
            plt.plot(o, 'og')  # plot using green circle markers
            plt.plot(t, 'or')  # plot using red circle markers
            plt.title(self.names[0])
            plt.pause(sys.float_info.epsilon)
