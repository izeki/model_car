from model_car.vis import *
import sys, traceback
import model_car.net_training.type_handlers.Bag_Folder as Bag_Folder

def preprocess_Bag_Folders(bag_folders_path_meta_path,bag_folders_path_rgb1to4_path,NUM_STATE_ONE_STEPS=30,graphics=False,accepted_states=[1],pkl_name='Bag_Folder.pkl'):
    
    
    bag_folders_paths_list = sorted(gg(opj(bag_folders_path_meta_path,'*')),key=natural_keys)

    
    for bfp in bag_folders_paths_list:

        try:

            print bfp
            run_name = bfp.split('/')[-1]

            left_image_bound_to_data_name = get_preprocess_dir_name_info(bfp)
            if left_image_bound_to_data_name == None:
                cprint("if left_image_bound_to_data_name == None:",'red')
                continue

            if len(gg(opj(bfp,pkl_name))) == 1:
                print('\t exists')
                if False: #graphics:
                    cprint(opj(run_name,'Bag_Folder.pkl')+' exists, loading it.','yellow','on_red')
                    BF = load_obj(opj(bfp,'Bag_Folder.pkl'))
            else:
                BF = Bag_Folder.init(bfp,
                    opj(bag_folders_path_rgb1to4_path,fname(bfp)),
                    left_image_bound_to_data_name=left_image_bound_to_data_name,
                    NUM_STATE_ONE_STEPS=NUM_STATE_ONE_STEPS,
                    accepted_states=accepted_states)
                if BF != None:
                    save_obj(BF,opj(bfp,'Bag_Folder.pkl'))

            if graphics:
                figure(run_name+' timecourses')
                plot(BF['data']['raw_timestamps'],100*BF['data']['encoder'],'y')
                plot(BF['data']['raw_timestamps'],BF['data']['state_one_steps'],'bo-')
                plot(BF['data']['good_start_timestamps'],zeros(len(BF['data']['good_start_timestamps']))+100,'go')
                plot(BF['data']['raw_timestamps'],2000*BF['data']['raw_timestamp_deltas'],'r')
                ylim(0,1000)

                figure(run_name+' raw_timestamp_deltas')
                rtd = BF['data']['raw_timestamp_deltas'].copy()
                rtd[rtd>0.08] = 0.08
                hist(rtd)
                #plot(BF['data']['raw_timestamps'],100*BF['data']['state'],'r')
                #plot(BF['data']['raw_timestamps'],100*BF['data']['acc_z'],'r')

                figure(run_name+' scatter')
                plot(BF['data']['steer'][BF['data']['good_start_indicies']],BF['data']['gyro_x'][BF['data']['good_start_indicies']],'o')

                plt.pause(0.001)

        except Exception as e:
            cprint("********** Exception ***********************",'red')
            traceback.print_exc(file=sys.stdout)
            #print(e.message, e.args)            



def get_preprocess_dir_name_info(bfp):
    fl = sgg(opj(bfp,'left*'))
    if len(fl) > 0:
        return sgg(opj(bfp,'left*'))[-1]
    else:
        return None
