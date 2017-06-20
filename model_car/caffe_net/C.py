# TO DO: Check topics for consistency

from model_car.vis import *
import threading
import caffe
import model_car.data_analysis.data_parsing.access_bag_files as access_bag_files
from model_car.caffe_net.Caffe_Net import *



USE_GPU = True
if USE_GPU:
    caffe.set_device(0)
    caffe.set_mode_gpu()


            
loaded_bag_files_names = {}
played_bagfile_dic = {}
used_timestamps = {}

if True:
    BF_dic,BF_dic_keys_weighted = access_bag_files.load_Bag_Folders(opjD('bair_car_data_meta'),opjD('bair_car_data_rgb_1to4'))
    threading.Thread(target=access_bag_files.bag_file_loader_thread,args=(BF_dic,BF_dic_keys_weighted,5*60,loaded_bag_files_names,played_bagfile_dic)).start()
    while len(loaded_bag_files_names) < 1:
        cprint(d2s("""len(loaded_bag_files_names) =""",len(loaded_bag_files_names)))
        time.sleep(5)


#caffe_net = Caffe_Net.Caffe_Net(solver_file_path,'version 1','this one',weights_file_path)
#caffe_net = Caffe_Net.Caffe_Net(solver_file_path,'version 1','most recent',weights_file_path)
#caffe_net = Caffe_Net.Caffe_Net(solver_file_path,'version 1',None,None)


thread_id = 'caffe net 1'
solver_file_path = opjh("kzpy3/caf5/z2_color/solver1.prototxt")
version = 'version 1'
weights_file_mode = 'most recent'
weights_file_path = opjD('z2_color')
command_dic = {}
command_dic[thread_id] = 'start thread' # command_dic[thread_id] = 'stop thread'
gpu_num = 1

def caffe_net_thread(thread_id,solver_file_path,version,weights_file_mode,weights_file_path,command_dic,gpu_num):
    caffe.set_device(gpu_num)
    caffe.set_mode_gpu()

    caffe_net = Caffe_Net(solver_file_path,version,weights_file_mode,weights_file_path)

    cprint(d2s('Starting thread ',thread_id),'yellow','on_blue')
    state = command_dic[thread_id]
    while True:
        command = command_dic[thread_id]
        if command == 'pause thread':
            if state == 'pause':
                pass
            else:
                state = 'pause'
                cprint(d2s('Pausing thread ',thread_id),'yellow','on_blue')
            time.sleep(1)
            continue
        if command == 'start thread' and state == 'pause':
            state = 'running'
            cprint(d2s('Unpausing thread ',thread_id),'yellow','on_blue')            
        if command == 'stop thread':
            cprint(d2s('Stopping thread ',thread_id),'yellow','on_blue')
            return
        if True: #try:
            data = access_bag_files.get_data(BF_dic,played_bagfile_dic,used_timestamps)
        else: #except Exception as e:
            cprint("********** Exception ***********************",'red')
            print(e.message, e.args)
            
        if data != None:
            caffe_net.train_step(data)
"""
threading.Thread(target=caffe_net_thread,args=(thread_id,solver_file_path,version,weights_file_mode,weights_file_path,command_dic,gpu_num)).start()
"""
caffe_net = Caffe_Net(solver_file_path,version,weights_file_mode,weights_file_path)

while True:
    try:
        data = access_bag_files.get_data(BF_dic,played_bagfile_dic,used_timestamps)
    except Exception as e:
        cprint("********** Exception ***********************",'red')
        print(e.message, e.args)
    if data != None:
        caffe_net.train_step(data)

"""
caffe_net = Caffe_Net.Caffe_Net(solver_file_path,version,weights_file_mode,weights_file_path)
def caffe_net_thread():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    while True:
        try:
            data = access_bag_files.get_data(BF_dic,played_bagfile_dic,used_timestamps)
        except Exception as e:
            cprint("********** Exception ***********************",'red')
            print(e.message, e.args)
            
        if data != None:
            caffe_net.train_step(data)
threading.Thread(target=caffe_net_thread).start()
"""
while False:
    try:
        data = access_bag_files.get_data(BF_dic,played_bagfile_dic,used_timestamps)
    except Exception as e:
        cprint("********** Exception ***********************",'red')
        print(e.message, e.args)
        
    if data != None:
        caffe_net.train_step(data)




if False:
    if command_file_timer.check():
        command_file_timer.reset()
        command_file_str_lst = txt_file_to_list_of_strings(command_file)
        command_str = ''
        for c in command_file_str_lst:
            command_str += c + '\n'
        print command_str
