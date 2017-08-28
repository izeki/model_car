"""Handles data loading."""
import random

import libs.type_handlers.Segment_Data as Segment_Data
from libs.utils2 import opjD, lo
from Parameters import ARGS


class DataIndex:
    """
    Index object, keeps track of position in data stack.
    """

    def __init__(self, valid_data_moments, ctr, epoch_counter):
        self.valid_data_moments = valid_data_moments
        self.ctr = ctr
        self.epoch_counter = epoch_counter
        self.epoch_complete = False


class Data:
    def __init__(self):

        # Load hdf5 segment data
        self.hdf5_runs_path = self.hdf5_segment_metadata_path = ARGS.data_path
        self.hdf5_runs_path += '/hdf5/runs'
        self.hdf5_segment_metadata_path += '/hdf5/segment_metadata'

        Segment_Data.load_Segment_Data(self.hdf5_segment_metadata_path,
                                       self.hdf5_runs_path)

        # Load data indexes for training and validation
        train_all_steer_path = ARGS.data_path + '/train_all_steer'
        val_all_steer_path = ARGS.data_path + '/val_all_steer'
        print('loading train_valid_data_moments...')
        self.train_index = DataIndex(lo(train_all_steer_path), -1, 0)
        print('loading val_valid_data_moments...')
        self.val_index = DataIndex(lo(val_all_steer_path), -1, 0)

    @staticmethod
    def get_data(run_code, seg_num, offset):
        data = Segment_Data.get_data(run_code, seg_num, offset,
                                     ARGS.stride * ARGS.nsteps, offset,
                                     ARGS.nframes, ignore=ARGS.ignore,
                                     require_one=ARGS.require_one,
                                     use_states=ARGS.use_states)
        return data

    @staticmethod
    def next(data_index):
        if data_index.ctr >= len(data_index.valid_data_moments) - (
                1 + ARGS.batch_size):  # Skip last batch if it runs out of data
            data_index.ctr = -1
            data_index.epoch_counter += 1
            data_index.epoch_complete = True
        if data_index.ctr == -1:
            data_index.ctr = 0
            print('shuffle start')
            random.shuffle(data_index.valid_data_moments)
            print('shuffle finished')
        data_index.ctr += 1
        return data_index.valid_data_moments[data_index.ctr]
