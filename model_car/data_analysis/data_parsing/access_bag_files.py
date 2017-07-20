# TO DO: Check topics for consistency

from model_car.vis import *
import model_car.net_training.type_handlers.Bag_Folder as Bag_Folder
import model_car.net_training.type_handlers.Bag_File as Bag_File
from model_car.net_training.type_handlers.preprocess_Bag_Folders import get_preprocess_dir_name_info as get_preprocess_dir_name_info
import cv2
import sys, traceback


NUM_STATE_ONE_STEPS = 30

thread_please_load_data = True # thread_please_load_data = False
bag_file_loader_thread_please_exit = False

def load_Bag_Folders(bag_folders_path_meta_path,bag_folders_rgb_1to4_path):
    BF_dic = {}
    BF_dic_keys_weights = []
    bag_folders_paths_list = sorted(gg(opj(bag_folders_path_meta_path,'*')),key=natural_keys)

    ctr = 0
    for bfp in bag_folders_paths_list:
        

        if False: #ctr > 3:
            print "Temp, returning"
            return BF_dic
        num_bag_files = len(gg(opj(bag_folders_path_meta_path.replace('meta','rgb_1to4'),'*')))
        run_name = fname(bfp)
        for i in range(int(num_bag_files/10)):
            BF_dic_keys_weights.append(run_name)
        left_image_bound_to_data_name = get_preprocess_dir_name_info(bfp)
        #cprint(opj(bfp,'Bag_Folder.pkl'),'blue','on_green')
        if len(gg(opj(bfp,'Bag_Folder.pkl'))) == 1:
            print('')
            cprint(opj(run_name,'Bag_Folder.pkl')+' exists, loading it.','yellow','on_red')
            BF = load_obj(opj(bfp,'Bag_Folder.pkl'))
        else:
            cprint('ERROR!!! '+opj(run_name,'Bag_Folder.pkl')+' does not exist!','yellow','on_red')
            continue
            assert(False)

        if run_name in BF_dic:
            cprint('ERROR!!! '+run_name+' already in BF_dic!','yellow','on_red')
            assert(False)
        BF_dic[run_name] = BF
        ctr += 1
    return BF_dic,BF_dic_keys_weights


def bag_file_loader_thread(BF_dic,BF_dic_keys_weights,delay_before_delete,loaded_bag_files_names,played_bagfile_dic): 

    while True:
        if bag_file_loader_thread_please_exit:
            cprint('THREAD:: exiting bag_file_loader_thread()')
            return
        elif not thread_please_load_data:
            time.sleep(1)
        else:
            if len(loaded_bag_files_names) > 1000:
                cprint('\n\nTHREAD:: pause before deleting '+bf+'\n\n,,','blue','on_red')
                time.sleep(delay_before_delete)

                played_bagfile_dic_keys = []
                played_bagfile_dic_values = []
                for b in played_bagfile_dic.keys():
                    played_bagfile_dic_keys.append(b)
                    played_bagfile_dic_values.append(played_bagfile_dic[b])
                indicies = [i[0] for i in sorted(enumerate(played_bagfile_dic_values),key=lambda x:x[1])]
                indicies.reverse()
                ctr = 0
                for i in indicies:
                    if ctr >= 25:
                        break
                    bf = played_bagfile_dic_keys[i]
                    if bf in loaded_bag_files_names:
                        #bf = a_key(loaded_bag_files_names)
                        cprint('THREAD:: deleting '+bf,'blue','on_red')
                        r = loaded_bag_files_names[bf]
                        loaded_bag_files_names.pop(bf)
                        BF = BF_dic[r]
                        BF['bag_file_image_data'].pop(bf)
                        ctr += 1
            if True: #try:
                r = random.choice(BF_dic.keys())
                BF = BF_dic[r]
                if type(BF) != dict:
                    continue
                dic_keys = ['bag_file_image_data','good_bag_timestamps','binned_timestamps','binned_steers','bid_timestamps']
                for dk in dic_keys:
                    if dk not in BF:
                        BF[dk] = {}
                #print BF['bag_file_num_dic']
                if len(BF['bag_file_num_dic']) > 0:
                    try:
                        bf = random.choice(BF['bag_file_num_dic'])
                    except:
                        continue
                    if bf in BF['bag_file_image_data']:
                        continue
                    BF['bag_file_image_data'][bf] = Bag_File.load_images(bf,color_mode="rgb8",include_flip=True)
                    loaded_bag_files_names[bf] = r

                    bid = BF['bag_file_image_data'][bf]

                    bag_left_timestamps = sorted(bid['left'].keys())

                    good_bag_timestamps = list(set(BF['data']['good_start_timestamps']) & set(bag_left_timestamps))
                    
                    #cprint(d2s('THREAD:: ',bf.split('/')[-1],'len(good_bag_timestamps) =',len(good_bag_timestamps)),'blue')

                    binned_timestamps = [[],[]]
                    binned_steers = [[],[]]

                    for t in good_bag_timestamps:
                        steer = BF['left_image_bound_to_data'][t]['steer']
                        if steer < 43 or steer > 55:
                            binned_timestamps[0].append(t)
                            binned_steers[0].append(steer)
                        else:
                            binned_timestamps[1].append(t)
                            binned_steers[1].append(steer)

                    BF['good_bag_timestamps'][bf] = good_bag_timestamps
                    BF['binned_timestamps'][bf] = binned_timestamps
                    BF['binned_steers'][bf] = binned_steers
                    BF['bid_timestamps'][bf] = sorted(bid['left'].keys())

            else: #except Exception as e:
                cprint("THREAD:: ********** Exception ***********************",'red')
                traceback.print_exc(file=sys.stdout)
                #print(e.message, e.args)

verbose = False
save_get_data_timer = Timer(60)

def get_data(BF_dic,played_bagfile_dic,used_timestamps):
    data = {}
    r = random.choice(BF_dic.keys())
    BF = BF_dic[r]
    if type(BF) != dict:
        return None
    if 'bag_file_image_data' not in BF:
        #print("""if 'bag_file_image_data' not in BF:""")
        #time.sleep(1)
        return None
    if len(BF['bag_file_image_data']) < 1:
        #print("""if len(BF['bag_file_image_data']) < 1:""")
        #time.sleep(1)
        return None
    bf = a_key(BF['bag_file_image_data'])
    if bf not in BF['good_bag_timestamps']:
        cprint(bf + """ not in BF['good_bag_timestamps']""")
        return None
    if len(BF['good_bag_timestamps'][bf]) < 100:
        if verbose:
            print(d2s('MAIN:: skipping',bf.split('/')[-1],"len(good_bag_timestamps) < 100"))
        return None
    bid = BF['bag_file_image_data'][bf]
    if bf not in played_bagfile_dic:
        played_bagfile_dic[bf] = 0
    played_bagfile_dic[bf] += 1

    if len(BF['binned_timestamps'][bf][0]) > 0 and len(BF['binned_timestamps'][bf][1]) > 0:
        rt = random.choice(BF['binned_timestamps'][bf][np.random.randint(2)])
    elif len(BF['binned_timestamps'][bf][0]) > 0:
        rt = random.choice(BF['binned_timestamps'][bf][0])
    elif len(BF['binned_timestamps'][bf][1]) > 0:
        rt = random.choice(BF['binned_timestamps'][bf][1])
    else:
        return None        

    topics = ['left','right','left_flip','right_flip','steer','motor','state','gyro_x','gyro_y','gyro_z','gyro_yz_mag']
    for tp in topics:
        data[tp] = []

    ts = BF['bid_timestamps'][bf]
    for i in range(len(ts)):
        if ts[i] == rt:
            if len(ts) > i+NUM_STATE_ONE_STEPS:
                if rt not in used_timestamps:
                    used_timestamps[rt] = 0
                used_timestamps[rt] += 1
                if save_get_data_timer.check():
                    save_obj(played_bagfile_dic,opjD('played_bagfile_dic'))
                    save_obj(used_timestamps,opjD('used_timestamps'))
                    save_get_data_timer.reset()
                for j in range(i,i+NUM_STATE_ONE_STEPS):
                    t = ts[j]

                    steer = BF['left_image_bound_to_data'][t]['steer']
                    motor = BF['left_image_bound_to_data'][t]['motor']
                    #state = BF['left_image_bound_to_data'][t]['state']
                    gyro_x = BF['left_image_bound_to_data'][t]['gyro'][0]
                    gyro_y = BF['left_image_bound_to_data'][t]['gyro'][1]
                    gyro_z = BF['left_image_bound_to_data'][t]['gyro'][2]
                    gyro_yz_mag = np.sqrt(gyro_y**2+gyro_z**2)
                    img_left = bid['left'][t]
                    right_timestamp = BF['left_image_bound_to_data'][t]['right_image']
                    img_right = bid['right'][right_timestamp]

                    img_left_flip = bid['left_flip'][t]
                    img_right_flip = bid['right_flip'][right_timestamp]

                    data['path'] = bf
                    data['steer'].append(steer)
                    data['motor'].append(motor)
                    data['gyro_x'].append(gyro_x)
                    data['gyro_y'].append(gyro_y)
                    data['gyro_z'].append(gyro_z)
                    data['gyro_yz_mag'].append(gyro_yz_mag)
                    data['left'].append(img_left)
                    data['right'].append(img_right)
                    data['left_flip'].append(img_left_flip)
                    data['right_flip'].append(img_right_flip)

                break
            else:
                if verbose:
                    cprint("MAIN:: ERROR!!!!!!!!! if len(ts) > i+10: failed")
    return data









