"""Utility classes for training."""
import os
import operator
import time
from Parameters import ARGS
from libs.utils2 import Timer, d2s
from libs.vis2 import mi
import matplotlib.pyplot as plt
import numpy as np


class MomentCounter:
    """Notify after N Data Moments Passed"""

    def __init__(self, n):
        self.start = 0
        self.n = n

    def step(self, data_index):
        if data_index.ctr - self.start >= self.n:
            self.start = data_index.ctr
            return True
        return False


class LossLog:
    """Keep Track of Loss, can be used within epoch or for per epoch."""

    def __init__(self):
        self.log = []
        self.ctr = 0
        self.total_loss = 0

    def add(self, ctr, loss):
        self.log.append((ctr, loss))
        self.total_loss += loss
        self.ctr += 1

    def average(self):
        return self.total_loss / (self.ctr * 1.)

    def export_csv(self, filename):
        np.savetxt(
            filename,
            np.array(self.log),
            header='Counter,Loss',
            delimiter=",",
            comments='')


class RateCounter:
    """Calculate rate of process in Hz"""

    def __init__(self):
        self.rate_ctr = 0
        self.rate_timer_interval = 10.0
        self.rate_timer = Timer(self.rate_timer_interval)

    def step(self):
        self.rate_ctr += 1
        if self.rate_timer.check():
            print('rate = ' + str(ARGS.batch_size * self.rate_ctr /
                                  self.rate_timer_interval) + 'Hz')
            self.rate_timer.reset()
            self.rate_ctr = 0


def save_net(weights_file_name, net, snap=False):
    if not snap:
        net.save(
            os.path.join(
                ARGS.save_path,
                weights_file_name
                + '.hdf5')
    else:
        net.save(
            os.path.join(
                ARGS.save_path,
                weights_file_name
                + '_snap.hdf5')
    


def display_sort_data_moment_loss(data_moment_loss_record, data):
    sorted_data_moment_loss_record = sorted(data_moment_loss_record.items(),
                                            key=operator.itemgetter(1))
    low_loss_range = range(20)
    high_loss_range = range(-1, -20, -1)

    for i in low_loss_range + high_loss_range:
        l = sorted_data_moment_loss_record[i]
        run_code, seg_num, offset = sorted_data_moment_loss_record[i][0][0]
        t = sorted_data_moment_loss_record[i][0][1]
        o = sorted_data_moment_loss_record[i][0][2]

        sorted_data = data.get_data(run_code, seg_num, offset)
        plt.figure(22)
        plt.clf()
        plt.ylim(0, 1)
        plt.plot(t, 'r.')
        plt.plot(o, 'g.')
        plt.plot([0, 20], [0.5, 0.5], 'k')
        mi(sorted_data['right'][0, :, :], 23, img_title=d2s(l[1]))
        plt.pause(1)
