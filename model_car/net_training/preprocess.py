import sys
sys.path.append('~/git/model_car/model_car')
from model_car.vis import *
from model_car.utils import *
import rospy
import rosbag
from collections import defaultdict
from model_car.data_analysis.data_parsing.Bagfile_Handler import Bagfile_Handler
from model_car.net_training.type_handlers.preprocess_bag_data import preprocess_bag_data
from model_car.net_training.type_handlers.preprocess_Bag_Folders import preprocess_Bag_Folders
import model_car.net_training.type_handlers.Bag_Folder as Bag_Folder
import model_car.net_training.type_handlers.Bag_File as Bag_File

"""
The goal here is to take a folder of bag files and bind all data to the
left ZED image timestamps. This involes interpolating the sensor data and
identifiying the correct right image which lags the left by ~5 ms. 

$ python preprocess.py /media/ubuntu/rosbags ~/Desktop/temp_bag 30
The first argument is source folder path for the bag files  . The second 
argument is the destination folder path for the pkl files. The third argument
is the frame rate for testing the quality of the synchronization (default is 30 fps). 
The preprocessed data is saved in the bag folder under

    .preprocessed2/left_image_bound_to_data.pkl

and

    .preprocessed2/preprocessed_data.pkl

which is simply all the original timestamps and sensor readings but no image data.
data preprocess



"""

def main(bag_folders_src_location, bag_folders_dst, NUM_STATE_ONE_STEPS):
    bag_folders_src = opj(bag_folders_src_location,'new' )
    bag_folders_dst_rgb1to4_path = opj(bag_folders_dst,'rgb_1to4')
    bag_folders_dst_meta_path = opj(bag_folders_dst,'meta')
    print('bag_folders_src: '+bag_folders_src+'; bag_folders_dst: '+bag_folders_dst+'; bag_folders_dst_meta_path: '+bag_folders_dst_meta_path+'; bag_folders_dst_rgb1to4_path: '+bag_folders_dst_rgb1to4_path)
    runs = sgg(opj(bag_folders_src,'*'))
    assert(len(runs) > 0)
      
    tb = '\t'
      
    cprint('Preliminary check of '+bag_folders_src)
    cprint("	checking bag file sizes and run durations")
      
    for r in runs:
        bags = sgg(opj(r,'*.bag'))
        print(bags)
        cprint(d2s(tb,fname(r),len(bags)))
        mtimes = []

        # for b in bags:
        #     bag_size = os.path.getsize(b)
        #     mtimes.append(os.path.getmtime(b))
        #     if bag_size < 0.99 * 1074813904:
        #         cprint(d2s('Bagfile',b,'has size',bag_size,'which is below full size.'),'red')
        #         unix('mv '+b+' '+b+'.too_small')

        #
        # mtimes = sorted(mtimes)
        # print(mtimes)
        # run_duration = mtimes[-1]-mtimes[0]
        # print(run_duration)
        # assert(run_duration/60./60. < 3.) # If clock set incorrectly, this can change during run leading to year-long intervals
        # cprint(d2s(r,'is okay'))
        #
    for r in runs:
        preprocess_bag_data(r)


    # The following code creates the rgb 1 to 4 folders. 
    # It takes a folder with runs in a "new" folder. 
    Bag_File.bag_folders_transfer_meta(bag_folders_src,bag_folders_dst_meta_path)
    Bag_File.bag_folders_save_images(bag_folders_src,bag_folders_dst_rgb1to4_path)


    graphics=False
    accepted_states=[1,3,5,6,7]
    pkl_name='Bag_Folder.pkl' # if different from 'Bag_Folder.pkl', (e.g., 'Bag_Folder_90_state_one_steps.pkl') files will be reprocessed.

    preprocess_Bag_Folders(bag_folders_dst_meta_path,
        bag_folders_dst_rgb1to4_path
        ,NUM_STATE_ONE_STEPS=NUM_STATE_ONE_STEPS,
        graphics=graphics,accepted_states=accepted_states,
        pkl_name=pkl_name)

    #os.rename(bag_folders_src,opj(bag_folders_src_location,'processed'))


if __name__ == "__main__":
    bag_folders_src_location = sys.argv[1]
    bag_folders_dst = sys.argv[2]
    NUM_STATE_ONE_STEPS = int(sys.argv[3])
    assert(is_number(NUM_STATE_ONE_STEPS))
    main(bag_folders_src_location, bag_folders_dst, NUM_STATE_ONE_STEPS)
