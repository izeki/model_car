import sys
import traceback
import cv2
import h5py

from training_data_generator import training_data_generator, visualize_data_model
from Parameters import ARGS
from libs.utils2 import *
from libs.vis2 import *
import libs.type_handlers.Bag_File as Bag_File


version = ARGS.version
verbose = ARGS.verbose
working_path = ARGS.data_path
dst_path = ARGS.data_path
NUM_STATE_ONE_STEPS = ARGS.nstateframes
meta_dir = os.path.join(working_path,'meta')
rgb_1to4_dir = os.path.join(working_path,'rgb_1to4')


def get_bag_names_dic(
        meta_dir,
        rgb_1to4_dir,
        to_ignore = ['xxx'],
        to_require=['']):#['ai','home','racing']):

    _,all_run_names = dir_as_dic_and_list(meta_dir)


    run_names_dic = {}
    
    bag_names_dic = {}


    for r in all_run_names:
        if (str_contains_one(r,to_ignore)) or (not str_contains_one(r,to_require)):
            cprint("Ignoring "+r,'yellow','on_blue')
        else:
            run_names_dic[r] = None

    for r in run_names_dic:
        p = opj(rgb_1to4_dir,r)
        if len(gg(p)) > 0:
            _,b = dir_as_dic_and_list(p)
            run_names_dic[r] = b
            for f in run_names_dic[r]:
                bag_names_dic[opj(r,f)] = False
        else:
            cprint("Warning: "+p+" does not exist",'red','on_cyan')

    num_bag_files_to_load = 0
    for r in sorted(run_names_dic):
        n = len(run_names_dic[r])
        cprint(d2n('\t',r,': ',n))
        num_bag_files_to_load += n
    cprint(d2s('All runs:',len(all_run_names),'Using runs:',len(run_names_dic)))
    cprint(d2n('Number of bags:',num_bag_files_to_load,', estimated hours = ',dp(num_bag_files_to_load/120.,1)))

    return bag_names_dic


def load_bag_file(bag_path,BagFolder_dic,bag_img_dic,skip_bag_dic,bag_names_dic,meta_dir,rgb_1to4_dir):

    bn = bag_path
    run_name = bn.split('/')[0]
    if bn in skip_bag_dic:
        print(d2n('\t',bn,' in skip_bag_dic'))
        return False
    bf = fname(bn)
    if bag_names_dic[bn] == False:
        #cprint(BagFolder_dic.keys(),'blue')
        if run_name not in BagFolder_dic:
            cprint('loading '+opj(run_name,'Bag_Folder.pkl'),'yellow','on_red')
            BagFolder_dic[run_name] = load_obj(opj(meta_dir,run_name,'Bag_Folder.pkl'))
        bag_img_dic[bn] = Bag_File.load_images(opj(rgb_1to4_dir,bn),color_mode="rgb8",include_flip=True)
        bag_names_dic[bn] == True

        good_bag_timestamps = list(set(BagFolder_dic[run_name]['data']['good_start_timestamps']) & set(bag_img_dic[bn]['left'].keys()))
        if len(good_bag_timestamps) < 100:
            if verbose:
                print(d2n('\t',bn,' len(good_bag_timestamps) < 100'))
            skip_bag_dic[bn] = True
            #skip_bag_dic[run_name] = True
            #del bag_img_dic[bn]
            #del bag_names_dic[bn]
            #del BagFolder_dic[run_name]
            return False
        binned_timestamps = [[],[]]
        binned_steers = [[],[]]

        for t in good_bag_timestamps:
            steer = BagFolder_dic[run_name]['left_image_bound_to_data'][t]['steer']
            if steer < 43 or steer > 55:
                binned_timestamps[0].append(t)
                binned_steers[0].append(steer)
            else:
                binned_timestamps[1].append(t)
                binned_steers[1].append(steer)
        dic_keys = ['bag_file_image_data','good_bag_timestamps','binned_timestamps','binned_steers','bid_timestamps']
        for dk in dic_keys:
            if dk not in BagFolder_dic[run_name]:
                BagFolder_dic[run_name][dk] = {}
        BagFolder_dic[run_name]['good_bag_timestamps'][bf] = good_bag_timestamps
        BagFolder_dic[run_name]['binned_timestamps'][bf] = binned_timestamps
        BagFolder_dic[run_name]['binned_steers'][bf] = binned_steers
        BagFolder_dic[run_name]['bid_timestamps'][bf] = sorted(bag_img_dic[bn]['left'].keys())            
        
        return True

    assert(False)
    

def get_data(run_name,bf,rt,BagFolder_dic,bag_img_dic,skip_bag_dic,NUM_STATE_ONE_STEPS):
    #print '***************',run_name,bf,rt
    try:
        data = {}

        if run_name in skip_bag_dic:
            return None

        
        bn = opj(run_name,bf)

        BF = BagFolder_dic[run_name]

        if type(BF) != dict:
            skip_bag_dic[bn] = True; return None

        if fname(bn) not in BF['good_bag_timestamps']:
            cprint(bf + """ not in BF['good_bag_timestamps']""")
            skip_bag_dic[bn] = True; return None
        #cprint(d2s("""len(BF['good_bag_timestamps'][fname(bn)]) =""",len(BF['good_bag_timestamps'][fname(bn)])),'yellow')
        if len(BF['good_bag_timestamps'][fname(bn)]) < 100:
            if verbose:
                print(d2s('MAIN:: skipping',bf.split('/')[-1],"len(good_bag_timestamps) < 100"))
            skip_bag_dic[bn] = True; return None

        bid = bag_img_dic[bn]

        topics = ['left','right','left_flip','right_flip','steer','motor','state']
        for tp in topics:
            data[tp] = []

        ts = BF['bid_timestamps'][bf]
        for i in range(len(ts)):
            if ts[i] == rt:
                if len(ts) > i+NUM_STATE_ONE_STEPS:

                    for j in range(i,i+NUM_STATE_ONE_STEPS):
                        t = ts[j]

                        steer = BF['left_image_bound_to_data'][t]['steer']
                        motor = BF['left_image_bound_to_data'][t]['motor']
                        #state = BF['left_image_bound_to_data'][t]['state']
                        #gyro_x = BF['left_image_bound_to_data'][t]['gyro'][0]
                        #gyro_y = BF['left_image_bound_to_data'][t]['gyro'][1]
                        #gyro_z = BF['left_image_bound_to_data'][t]['gyro'][2]
                        #gyro_yz_mag = np.sqrt(gyro_y**2+gyro_z**2)
                        img_left = bid['left'][t]
                        right_timestamp = BF['left_image_bound_to_data'][t]['right_image']
                        img_right = bid['right'][right_timestamp]

                        img_left_flip = bid['left_flip'][t]
                        img_right_flip = bid['right_flip'][right_timestamp]


                        data['path'] = bn
                        data['steer'].append(steer)
                        data['motor'].append(motor)
                        #data['gyro_x'].append(gyro_x)
                        #data['gyro_y'].append(gyro_y)
                        #data['gyro_z'].append(gyro_z)
                        #data['gyro_yz_mag'].append(gyro_yz_mag)
                        data['left'].append(img_left)
                        data['right'].append(img_right)
                        data['left_flip'].append(img_left_flip)
                        data['right_flip'].append(img_right_flip)


                    break
                else:
                    if verbose:
                        cprint("MAIN:: ERROR!!!!!!!!! if len(ts) > i+10: failed")

        try:
            if type(data['path']) != str:
                skip_bag_dic[bn] = True; return None
            for topic in ['steer','motor','left','right','left_flip','right_flip']:
                if type(data[topic]) != list:
                    skip_bag_dic[bn] = True; return None
                if len(data[topic]) != NUM_STATE_ONE_STEPS:
                    skip_bag_dic[bn] = True; return None
            
            for topic in ['steer','motor']:
                for v in data[topic]:
                    if not isinstance(v,(int,long,float)):
                        skip_bag_dic[bn] = True; return None

            for topic in ['left','right','left_flip','right_flip']:
                for v in data[topic]:
                    if not isinstance(v,np.ndarray):
                        skip_bag_dic[bn] = True; return None
            
        except:
            skip_bag_dic[bn] = True; return None
    except Exception as e:
        cprint("********** Exception ***********************",'red')
        traceback.print_exc(file=sys.stdout)
        #print(e.message, e.args)
        skip_bag_dic[bn] = True; return None
        
    return data



def visualize_data(data,dt=33,image_only=False):
    if not image_only:
        figure('steer motor')
        clf()
        #plt.subplot(1,3,3)
        plot((array(data['steer'])-49)/100.,'r')
        plot((array(data['motor'])-49)/100,'g')
        #plot((array(data['gyro_yz_mag'])/200.),'b')
        ylim(-0.5,0.5)
        pause(0.00001)
    for i in range(len(data['right'])):
        #mi(d['right'][i],'right')
        #pause(0.00000001)
        cv2.imshow('right',cv2.cvtColor(data['left'][i],cv2.COLOR_RGB2BGR))
        if cv2.waitKey(dt) & 0xFF == ord('q'):
            pass
    
# 
# 
# 
# if False:
#     import h5py
#     hdf5_filename = '/media/karlzipser/ExtraDrive1/solver_inputs.hdf5'
#     solver_inputs = h5py.File(hdf5_filename,'r')
#     ks = solver_inputs.keys()
#     while True:
#         k = random.choice(ks)
#         grp = solver_inputs[k]
#         for i in range(12):
#             mi(grp['ZED_data_pool2'][0,i,:,:],1)
#             print(grp['metadata'][:])
#             print(grp['steer_motor_target_data'][:])
#             pause(0.5)
#         #grp['metadata']
#         #grp['steer_motor_target_data']


def main():

    timer = Timer(30)

    bag_names_dic = get_bag_names_dic(meta_dir,rgb_1to4_dir,)
    bag_names_list = sorted(bag_names_dic,key=natural_keys)

    hdf5_runs_dic = {}

    if 'skip_bag_dic' not in locals():
        skip_bag_dic = {}
    skip_bag_dic = {}
    BagFolder_dic = {}
    
    ctr = 0
    t0 = time.time()

    previous_run_name = 'nothing'

    for bn in bag_names_list:
        bag_img_dic = {}
        load_bag_file(bn,BagFolder_dic,bag_img_dic,skip_bag_dic,bag_names_dic,meta_dir,rgb_1to4_dir)
        bf = fname(bn)

        run_name = bn.split('/')[0]
        
        #if run_name != previous_run_name:
        #    if previous_run_name != 'nothing':
        #        hdf5_runs_dic[previous_run_name].close()
        #    previous_run_name = run_name
        
        file_name = os.path.join(dst_path,'hdf5','runs',run_name + '.hdf5')
        unix('mkdir -p '+os.path.join(dst_path,'hdf5','runs'))
        if run_name not in hdf5_runs_dic:
            hdf5_runs_dic[run_name] = h5py.File(file_name)
        solver_inputs = hdf5_runs_dic[run_name]
        if run_name in BagFolder_dic:
            if 'good_bag_timestamps' in BagFolder_dic[run_name]:
                if bf in BagFolder_dic[run_name]['good_bag_timestamps']:
                    cprint(d2s("""len(BagFolder_dic[run_name]['good_bag_timestamps'][bf]=""",
                               len(BagFolder_dic[run_name]['good_bag_timestamps'][bf])))
                    binned_timestamps = BagFolder_dic[run_name]['binned_timestamps'][bf]
                    ln = int(1.0 * min(len(binned_timestamps[0]),len(binned_timestamps[1])))
                    random.shuffle(binned_timestamps[0]); random.shuffle(binned_timestamps[1])
                    timestamps = binned_timestamps[0][:]+binned_timestamps[1][:ln]
                    assert(len(timestamps) == len(binned_timestamps[0])+ln)
                    timestamps = sorted(timestamps)
                    for flip in [True,False]:
                        for ts in timestamps:
                            ctr += 1
                            #print ts
                            data = get_data(run_name,bf,ts,BagFolder_dic,bag_img_dic,skip_bag_dic,NUM_STATE_ONE_STEPS)
                            if data != None:
                                result = training_data_generator(version, data, flip, show_data=False, camera_dropout=True)
                                if result != None:
                                    n = d2f('-',bn.replace('/','-'),ts,ctr,flip)
                                    x_train = result['x_train']
                                    y_train = result['y_train']
                                    if n not in solver_inputs:
                                        grp = solver_inputs.create_group(n)
                                        grp['ZED_input'] = x_train['ZED_input'][:,:,:,:].astype('uint8')
                                        grp['meta_input'] = x_train['meta_input'][:]
                                        grp['steer_motor_target_data'] = y_train['steer_motor_target_data'][:]

                                if timer.check(): #mod(ctr,30)==0:#
                                    #solver_inputs.close()
                                    #solver_inputs = h5py.File(hdf5_filename)
                                    visualize_data(data)
                                    if result != None:
                                        visualize_data_model(version, result, flip)
                                    #ctr = 0
                                    cprint(d2s('ctr =',
                                               ctr,
                                               'rate =',
                                               dp(ctr / (time.time() - t0), 1),
                                               'Hz',
                                               'size =',
                                               dp(os.path.getsize(file_name) / 10 ** 12., 4),'TB'),
                                           'red',
                                           'on_yellow') 
                                    cprint(d2s('timestamp percent =', 
                                               100 * len(timestamps) / (1. * len(BagFolder_dic[run_name]['good_bag_timestamps'][bf]))),
                                           'green')
                                    timer.reset()
    #hdf5_runs_dic[previous_run_name].close()
    for r in hdf5_runs_dic.keys():
        try:
            cprint("----------closing file {} -------------------".format(r), 'red')
            hdf5_runs_dic[r].close()
        except Exception as e:
            cprint("********** Exception ***********************",'red')
            traceback.print_exc(file=sys.stdout)
                    
if __name__ == "__main__":
    main()
    print('------------------------------finished pkls to hdf5 conversion-----------------------')
            