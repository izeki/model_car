import sys 
import traceback
import rospy
import rosbag
from ..utils2 import *


############## topics, not necessarily original rosbag names ###################
#
image_topics = ['left_image','right_image']
single_value_topics = ['steer','state','motor','encoder']#,'GPS2_lat']
vector3_topics = ['acc','gyro','gps','gyro_heading']
all_topics = image_topics + single_value_topics + vector3_topics
#
######################################################################

############## bagfile data processing to useful forms ##############################
#
A = {} # this will be renamed preprocessed_data for return


def preprocess_bag_data(bag_folder_path,bagfile_range=[]):
    
    A = {} # this will be renamed preprocessed_data for return

    for topic in all_topics:
        A[topic] = {}

    bag_files = sorted(glob.glob(opj(bag_folder_path,'*.bag')))
    
    if len(bagfile_range) > 0:
        bag_files = bag_files[bagfile_range[0]:(bagfile_range[1]+1)]
    
    #pprint(bag_files)
    cprint(d2s('Processing',len(bag_files),'bag files:'),'red')
    for b in bag_files:
        cprint('\t'+b,'blue')

    for b in bag_files:
        try:
            cprint(b,'yellow')

            bag = rosbag.Bag(b)

            for topic in single_value_topics:
                for m in bag.read_messages(topics=['/bair_car/'+topic]):
                    t = round(m.timestamp.to_time(),3) # millisecond resolution
                    if not isinstance(m[1].data,(int,long,float)):
                        print("if not isinstance(m[1].data,(int,long,float)):")
                        print(d2s("m[1].data = ",m[1].data))
                        assert(False)
                    A[topic][t] = m[1].data

            for topic in vector3_topics:
                if topic != 'gps':
                    for m in bag.read_messages(topics=['/bair_car/'+topic]):
                        t = round(m.timestamp.to_time(),3)
                        if not isinstance(m[1].x,(int,long,float)):
                            print("if not isinstance(m[1].x,(int,long,float)):")
                            print(d2s("m[1].x = ",m[1].x))
                            assert(False)
                        if not isinstance(m[1].y,(int,long,float)):
                            print("if not isinstance(m[1].y,(int,long,float)):")
                            print(d2s("m[1].y = ",m[1].y))
                            assert(False)
                        if not isinstance(m[1].z,(int,long,float)):
                            print("if not isinstance(m[1].x,(int,long,float)):")
                            print(d2s("m[1].z = ",m[1].z)) 
                            assert(False)
                        A[topic][t] = (m[1].x,m[1].y,m[1].z)

            try:
                topic = 'gps'
                for m in bag.read_messages(topics=['/bair_car/'+topic]):
                    t = round(m.timestamp.to_time(),3)
                    A[topic][t] = (m[1].latitude,m[1].longitude,m[1].altitude)
            except:
                print 'gps problem'

            for m in bag.read_messages(topics=['/bair_car/zed/left/image_rect_color']):
                t = round(m.timestamp.to_time(),3)
                A['left_image'][t] = 'z' # placeholder for ctr

            for m in bag.read_messages(topics=['/bair_car/zed/right/image_rect_color']):
                t = round(m.timestamp.to_time(),3)
                A['right_image'][t] = 'z' # placeholder for ctr
        except Exception as e:
            cprint("********** Exception ***********************",'red')
            traceback.print_exc(file=sys.stdout)
            #print e.message, e.args


    for img in ['left_image','right_image']:
        ctr = 0
        sorted_timestamps = sorted(A[img].keys())
        for t in sorted_timestamps:
            A[img][t] = ctr
            ctr += 1
    
    preprocessed_data = A

    left_image_bound_to_data,error_log = _bind_left_image_timestamps_to_data(A)
    print """left_image_bound_to_data,error_log = _bind_left_image_timestamps_to_data(A) """

    if False:
        timestamps = sorted(left_image_bound_to_data.keys())
        state_one_steps = 0
        for i in range(len(timestamps)-1,-1,-1):
            if left_image_bound_to_data[timestamps[i]]['state'] == 1.0:
                state_one_steps += 1
            else:
                state_one_steps = 0
            left_image_bound_to_data[timestamps[i]]['state_one_steps'] = state_one_steps

    

    dst_path = opj(bag_folder_path,'.preprocessed2')
    print """unix('mkdir -p ' """ +dst_path+')'
    unix('mkdir -p '+dst_path)

    print """save_obj(left_image_bound_to_data,opj(dst_path,'left_image_bound_to_data')) """
    save_obj(left_image_bound_to_data,opj(dst_path,'left_image_bound_to_data'))

    print """save_obj(preprocessed_data,opj(dst_path,'preprocessed_data'))"""
    save_obj(preprocessed_data,opj(dst_path,'preprocessed_data'))


    return preprocessed_data,left_image_bound_to_data

#
######################################################################
# 
#
########################## binding data to left_image timestamps ######
#

def _bind_left_image_timestamps_to_data(A):

    ms_timestamps = {}

    ms_timestamps['right_image'] = _assign_right_image_timestamps(A)

    for topic in single_value_topics:
        try:
            ms_timestamps[topic] = _interpolate_single_values(A,topic)
        except:
            print 'Error with topic '+topic
    for topic in vector3_topics:
        try:
            ms_timestamps[topic] = _interpolate_vector_values(A,topic)
        except:
            print 'Error with topic '+topic

    left_image_bound_to_data = {}

    error_log = []

    sorted_keys = sorted(A['left_image'].keys())
    for i in range(30,len(sorted_keys)-30):
    # we throw away the first and last 5 frames to avoid boundry problems with other sensors
        k = sorted_keys[i]
        left_image_bound_to_data[k] = {}
        for l in ms_timestamps.keys():
            try:
                left_image_bound_to_data[k][l] = ms_timestamps[l][k]
            except:
                error_log.append((k,l))
                left_image_bound_to_data[k][l] = 'no data'
                print (k,l)
                cprint("""
            except:
                error_log.append((k,l))
                left_image_bound_to_data[k][l] = 'no data'
                print (k,l)                    
                    """,'red','on_blue')
    print error_log
    return left_image_bound_to_data,error_log


def _interpolate_single_values(A,topic):
    """
    Warning, this will interpolate the topic 'state', which we do not want.
    """
    interp_dic = {}
    k,d = get_sorted_keys_and_data(A[topic])
    for i in range(0,len(k)-1):
        for j in range(int(k[i]*1000),int(k[i+1]*1000)):
            v =  round((d[i+1]-d[i])/(k[i+1]-k[i]) * (j/1000.-k[i])  + d[i],3)
            interp_dic[j/1000.] = v
    return interp_dic

def _interpolate_vector_values(A,topic):
    interp_dic = {}
    k,d = get_sorted_keys_and_data(A[topic])
    d = np.array(d)
    dim = len(d[0])
    for i in range(0,len(k)-1):
        for j in range(int(k[i]*1000),int(k[i+1]*1000)):
            v = []
            for u in range(dim):
                if topic != 'gps': # with GPS we need as many decimal places as possible
                    v.append(  round((d[i+1,u]-d[i,u])/(k[i+1]-k[i]) * (j/1000.-k[i])  + d[i,u], 3))
                else:
                    v.append((d[i+1,u]-d[i,u])/(k[i+1]-k[i]) * (j/1000.-k[i])  + d[i,u])
            interp_dic[j/1000.] = v
    return interp_dic

def _assign_right_image_timestamps(A):
    interp_dic = {}
    k,d = get_sorted_keys_and_data(A['right_image'])
    for i in range(0,len(k)-1):
        a = int(k[i]*1000)
        b = int(k[i+1]*1000)
        c = (a+b)/2
        for j in range(a,b):
            if j < c:
                v = k[i]
            else:
                v = k[i+1]
            interp_dic[j/1000.] = v
    return interp_dic
#
######################################################################
