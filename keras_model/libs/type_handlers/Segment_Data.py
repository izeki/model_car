"""Helper functions to load segment data from runs."""
import sys
from ..progress import *
from ..vis2 import *


"""
This should be pure data, no AI code
"""

i_variables = [
    'state',
    'steer',
    'motor',
    'run_',
    'runs',
    'run_labels',
    'meta_path',
    'rgb_1to4_path',
    'B_',
    'left_images',
    'right_images',
    'unsaved_labels']

i_labels = [
    'mostly_caffe',
    'mostly_human',
    'aruco_ring',
    'out1_in2',
    'direct',
    'home',
    'furtive',
    'play',
    'racing',
    'multicar',
    'campus',
    'night',
    'Smyth',
    'left',
    'notes',
    'local',
    'Tilden',
    'reject_run',
    'reject_intervals',
    'snow',
    'follow',
    'only_states_1_and_6_good']

not_direct_modes = ['out1_in2','left','furtive','play','racing','follow']

for q in i_variables + i_labels:
    exec(d2n(q, ' = ', "\'", q, "\'")) 


Segment_Data = {} # main data dictionary for segments


def function_load_hdf5(path):
    """
    def function_load_hdf5(path):
    
    The hdf5 data files each contain data for a single data collection run
    which may be between five and 120 minutes in length.

    Data in the files is stored in 'segments' of good data. Good here includes
    data that has well behaved timestamps (no gaps) as well as data collected
    in data collection modes (i.e., not in mode 2 or 4).

    Each run also has 'labels' with information that applies to all segments.

    This function returns both:
    
            return (labels,S)
    """
    F = h5py.File(path)
    labels = {}
    Lb = F['labels']
    for k in Lb.keys():
        if Lb[k][0]:
            labels[k] = True
        else:
            labels[k] = False
    S = F['segments']
    return (labels, S)


def load_animate_hdf5(path, start_at_time=0):
    """
    def load_animate_hdf5(path,start_at_time=0):
            or
    A5(path)

    Utility to animate the segments of a run, with steer and motor data
    added to frames.

    start_at_time parameter allows for synchronizing animations
    across different command lines. It does not refer to time within the dataset.

    e.g.,
            from kzpy3.teg7.train_with_hdf5_utils import *
            load_animate_hdf5(opjD('bair_car_data/hdf5/runs/racing_Tilden_27Nov16_12h54m34s_Mr_Orange.hdf5'))
            load_animate_hdf5(opjD('bair_car_data/hdf5/runs/direct_caffe2_local_01Feb17_16h18m25s_Mr_Silver.hdf5'))
    Here the bars are blue, human control, and the motor is held at the maximum most of the time.
    """
    start_at(start_at_time)
    l, s = function_load_hdf5(path)
    img = False
    for h in range(len(s)):
        if not isinstance(img, bool):
            img *= 0
            img += 128
            mi_or_cv2(img)
        pause(0.5)
        n = str(h)
        for i in range(len(s[n]['left'])):
            img = s[n]['left'][i]
            bar_color = [0, 0, 0]
            if s[n][state][i] == 1:  # Full human control
                bar_color = [0, 0, 255]  # Blue

            elif s[n][state][i] == 6:  # Full AI model control
                bar_color = [255, 0, 0]  # Red

            elif s[n][state][i] == 3:  # Human motor, AI steer
                bar_color = [255, 128, 128]  # Pink

            elif s[n][state][i] == 5:  # AI motor, human steer
                bar_color = [255, 255, 0]  # Yellow

            elif s[n][state][i] == 7:  # Human motor, Human steer
                bar_color = [255, 0, 255]  # Purple
                
            else:
                print s[n][state][i]
                # black, show not be seen, indicates state 2 or 4.
                bar_color = [0, 0, 0] 
            if i < 2:
                smooth_steer = s[n][steer][i]
            else:
                smooth_steer = (s[n][steer][i] 
                                + 0.5*s[n][steer][i-1]
                                + 0.25*s[n][sssssteer][i-2])/1.75
            apply_rect_to_img(
                img,
                smooth_steer,
                0,
                99,
                bar_color,
                bar_color,
                0.9,
                0.1,
                center=True,
                reverse=True,
                horizontal=True)
            apply_rect_to_img(
                img,
                s[n][motor][i],
                0,
                99,
                bar_color,
                bar_color,
                0.9,
                0.1,
                center=True,
                reverse=True,
                horizontal=False)
            mi_or_cv2(img)

            
A5 = load_animate_hdf5


def load_run_codes(hdf5_segment_metadata_path):
    """
    def load_run_codes():
    Each run is given a code number. This function loads the numbers into Segment_Data
    """
    run_codes = load_obj(opj(hdf5_segment_metadata_path, 'run_codes.pkl'))
    Segment_Data['run_codes'] = run_codes
    Segment_Data['runs'] = {}
    for n in run_codes.keys():
        run_name = run_codes[n]
        Segment_Data['runs'][run_name] = {}
        Segment_Data['runs'][run_name]['run_code'] = n

        
def run_into_Segment_Data(
        run_code_num,
        hdf5_segment_metadata_path,
        hdf5_runs_path):
    """
    def run_into_Segment_Data(run_code_num):
    This function loads a given run's data into Segment_Data
    """
    run_name = Segment_Data['run_codes'][run_code_num]
    assert(run_name in Segment_Data['runs'])
    sys.stdout.write("\033[K")
    sys.stdout.write(' ' + run_name + '\r')
    labels, segments = function_load_hdf5(
        opj(hdf5_runs_path, run_name + '.hdf5'))
    high_steer = load_obj(opj(hdf5_segment_metadata_path,
                              run_name + '.high_steer_data_moments.pkl'))
    low_steer = load_obj(opj(hdf5_segment_metadata_path,
                             run_name + '.low_steer_data_moments.pkl'))
    state_hist_list = load_obj(
        opj(hdf5_segment_metadata_path, run_name + '.state_hist_list.pkl'))
    Segment_Data['runs'][run_name]['labels'] = labels
    Segment_Data['runs'][run_name]['segments'] = segments
    Segment_Data['runs'][run_name]['high_steer'] = high_steer
    Segment_Data['runs'][run_name]['low_steer'] = low_steer
    Segment_Data['runs'][run_name]['state_hist_list'] = state_hist_list
    return run_name


def animate_segment(run_code_num, seg_num):
    """
    def animate_segment(run_code_num,seg_num):
    Animate a data segment
    """
    run_name = Segment_Data['run_codes'][run_code_num]
    left_images = Segment_Data['runs'][run_name]['segments'][str(
        seg_num)]['left'][:]
    steers = Segment_Data['runs'][run_name]['segments'][str(
        seg_num)]['steer'][:]
    motors = Segment_Data['runs'][run_name]['segments'][str(
        seg_num)]['motor'][:]
    states = Segment_Data['runs'][run_name]['segments'][str(
        seg_num)]['state'][:]
    for i in range(shape(left_images)[0]):
        bar_color = [0, 0, 0]
        if states[i] == 1:
            bar_color = [0, 0, 255]
        elif states[i] == 6:
            bar_color = [255, 0, 0]
        elif states[i] == 5:
            bar_color = [255, 255, 0]
        elif states[i] == 7:
            bar_color = [255, 0, 255]
        else:
            bar_color = [0, 0, 0]
        if i < 2:
            smooth_steer = steers[i]
        else:
            smooth_steer = (steers[i] 
                            + 0.5*steers[i-1] 
                            + 0.25*steers[i-2]) / 1.75
        img = left_images[i, :, :, :]
        apply_rect_to_img(
            img,
            smooth_steer,
            0,
            99,
            bar_color,
            bar_color,
            0.9,
            0.1,
            center=True,
            reverse=True,
            horizontal=True)
        apply_rect_to_img(
            img,
            motors[i],
            0,
            99,
            bar_color,
            bar_color,
            0.9,
            0.1,
            center=True,
            reverse=True,
            horizontal=False)
        mi_or_cv2(img)


def get_data(
        run_code_num,
        seg_num,
        offset,
        slen,
        img_offset,
        img_slen,
        ignore=[
            left,
            out1_in2],
        require_one=[],
        smooth_steer=True,
        use_states=[
            1,
            5,
            6,
            7],
        no_images=False):
    """
    def get_data(run_code_num,seg_num,offset,slen,img_offset,img_slen,ignore=[left,out1_in2],require_one=[],smooth_steer=True):

    This is the function that delivers segment data to load into Caffe. A run, segment and offset from segement beginning are
    specified. If there are insufficient data following the offset, None is returned.
    """
    run_name = Segment_Data['run_codes'][run_code_num]
    labels = Segment_Data['runs'][run_name]['labels']
    for ig in ignore:
        if labels[ig]:
            # print "ignored "+ig
            return None
    require_one_okay = True
    if len(require_one) > 0:
        require_one_okay = False
        for ro in require_one:
            if labels[ro]:
                require_one_okay = True
    if not require_one_okay:        
        return None
    a = offset
    b = offset + slen
    ia = img_offset
    ib = img_offset + img_slen
    seg_num_str = str(seg_num)
    if not (
            b -
            a <= len(
            Segment_Data['runs'][run_name]['segments'][seg_num_str]['steer'][:])):
        return None
    if not (
            ib -
            ia <= len(
            Segment_Data['runs'][run_name]['segments'][seg_num_str]['steer'][:])):
        return None
    steers = Segment_Data['runs'][run_name]['segments'][seg_num_str]['steer'][a:b]
    if len(steers) != slen:
        return None
    motors = Segment_Data['runs'][run_name]['segments'][seg_num_str]['motor'][a:b]
    if len(motors) != slen:
        return None
    states = Segment_Data['runs'][run_name]['segments'][str(
        seg_num)]['state'][a:b]
    if len(states) != slen:
        return None
    # This is not comprehensive, but is faster than checking every step. Think
    # about this further . . .
    if states[0] not in use_states: 
        return None
    if no_images:
        left_images = None
        right_images = None
    else:
        left_images = Segment_Data['runs'][run_name]['segments'][seg_num_str]['left'][ia:ib]
        right_images = Segment_Data['runs'][run_name]['segments'][seg_num_str]['right'][ia:ib]
    if smooth_steer:
        for i in range(2, len(steers)):
            steers[i] = (3 / 6.) * steers[i] + (2 / 6.) * \
                steers[i - 1] + (1 / 6.) * steers[i - 2]
    data = {}
    data['name'] = run_name
    data['steer'] = steers
    data['motor'] = motors
    data['states'] = states
    data['left'] = left_images
    data['right'] = right_images
    data['labels'] = labels
    return data


def load_Segment_Data(hdf5_segment_metadata_path, hdf5_runs_path):
    load_run_codes(hdf5_segment_metadata_path)
    pb = ProgressBar(len(Segment_Data['run_codes']))
    ctr = 0
    print("doing run_into_Segment_Data...")
    for n in Segment_Data['run_codes'].keys():
        ctr += 1
        pb.animate(ctr)
        run_into_Segment_Data(n, hdf5_segment_metadata_path, hdf5_runs_path)
    sys.stdout.write("\033[K")
    pb.animate(len(Segment_Data['run_codes']) - 1)
    print()
    