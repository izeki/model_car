from model_car.net_training.type_handlers.preprocess_bag_data import *
from model_car.net_training.type_handlers.Bag_File import *
from model_car.progress import *
from model_car.vis import *
import os
import sys, traceback

"""
This

Interactive data viewer for model car project.

Change path with SP(), i.e., function_set_paths()

e.g., in ipython type:

from kzpy3.teg7.interactive import *

or from command line type:

python kzpy3/teg7/interactive.py

Then type:

function_visualize_run()

This will visualize run data.

Type:

AR(600,610)

This will animate 10s of data. Note, frames that are not considered data are in grayscale.

Type:

LR()

to list runs. The first number is simply a count (starting with 0), the second number
is the number of bag files in the run. A bag file is 1GB of raw data (much less here)
and take up about 30s, although this varies with image complexity.

To choose a new run (say, run 53), type:

function_set_run(53)
function_visualize_run()

Note that the prompt on the command line lists the current run. Note that run 0 is selected by default.

Now try:

AR(900,920)

This will show going from non-data to data.

Note, sometimes there is a gap in the frames, as in this example.
The program will report this and pause during this time.
Using the TX1 dev. board cleans this up dramatically.


These runs need to be processed correctly:
22[18] direct_rewrite_test_24Apr17_13h09m31s_Mr_Blue  22    X:True 
25[24] direct_rewrite_test_24Apr17_13h31m59s_Mr_Black  25    X:True 
30[60] direct_rewrite_test_24Apr17_14h39m17s_Mr_Orange  30    X:True 




"""

i_variables = ['state','steer','motor','run_','runs','run_labels','meta_path','rgb_1to4_path','Bag_Folder_Filename','left_images','right_images','unsaved_labels']

i_labels = ['mostly_caffe','mostly_human','aruco_ring','out1_in2','direct','home','furtive','play','racing','multicar','campus','night','Smyth','left','notes','local','Tilden','reject_run','reject_intervals','snow','follow','only_states_1_and_6_good']
not_direct_modes = ['out1_in2','left','furtive','play','racing','follow']

i_functions = ['function_close_all_windows','function_set_plot_time_range','function_set_label','function_current_run','function_help','function_set_paths','function_list_runs','function_set_run','function_visualize_run','function_animate','function_run_loop']
for q in i_variables + i_functions + i_labels:
    print d2n(q,' = ',"\'",q,"\'")
    exec(d2n(q,' = ',"\'",q,"\'")) # global_dict use leading underscore because this facilitates auto completion in ipython

i_label_abbreviations = {aruco_ring:'ar_r',mostly_human:'mH',mostly_caffe:'mC',out1_in2:'o1i2', direct:'D' ,home:'H',furtive:'Fu',play:'P',racing:'R',multicar:'M',campus:'C',night:'Ni',Smyth:'Smy',left:'Lf',notes:'N',local:'L',Tilden:'T',reject_run:'X',reject_intervals:'Xi',snow:'S',follow:'F',only_states_1_and_6_good:'1_6'}

global_dict = {}

#bair_car_data_path = opjD('bair_car_data_new')
#bair_car_data_path = '/media/karlzipser/ExtraDrive4/bair_car_data_new_28April2017'
#bair_car_data_path = sys.argv[1]
bair_car_data_path = None



def function_close_all_windows():
    plt.close('all')


def function_help():
    """
    function_help(q=None)
            HE
            get help.
    """
    cprint('INTERACTIVE FUNCTIONS:')
    for f in i_functions:
        exec('print('+f+'.__doc__)')
    cprint('INTERACTIVE VARIABLES:')
    tab_list_print(i_variables)
    cprint('\nINTERACTIVE LABELS:')
    tab_list_print(i_labels)


def blank_labels():
    l = {}
    l[local] = False
    l[Tilden] = False
    l[reject_run] = False
    l[reject_intervals] = False
    l[snow] = False
    l[follow] = False
    l[only_states_1_and_6_good] = False
    return l


def function_set_paths(p=opj(bair_car_data_path)):
    """
    function_set_paths(p=opj(bair_car_data_path))
        SP
    """
    global global_dict
    global_dict[meta_path] = opj(p,'meta')
    global_dict[rgb_1to4_path] = opj(p,'rgb_1to4')
    global_dict[runs] = sgg(opj(global_dict[meta_path],'*'))
    for j in range(len(global_dict[runs])):
        global_dict[runs][j] = fname(global_dict[runs][j])
    global_dict[run_] = global_dict[runs][0]
    cprint('meta_path = '+global_dict[meta_path])

# show the state distribution
def function_current_run():
    """
    function_current_run()
        function_current_run
    """
    r=global_dict[run_]
    n = len(gg(opj(global_dict[rgb_1to4_path],r,'*.bag.pkl')))
    cprint(d2n('[',n,'] ',r))
    state_hist = np.zeros(10)
    left_images=global_dict[Bag_Folder_Filename]['left_image_bound_to_data']
    for left_image in left_images:
        s = left_images[left_image]['state']
        if type(s) == str:
            s = 0
        else:
            s = int(s)
        state_hist[s]+=1
    state_hist /= state_hist.sum()
    state_percent = []
    for i in range(0,8):
        s = state_hist[i]
        state_percent.append(int(100*s))
    print(d2s('State percentges:',state_percent[1:8]))
    print(global_dict[run_labels][r])


def function_list_runs(rng=None,auto_direct_labelling=True):
    """
    function_list_runs()
        LR
    """
    cprint(global_dict[meta_path])
    try:
        run_labels_path = most_recent_file_in_folder(opj(bair_car_data_path,'run_labels'),['run_labels'])
        global_dict[run_labels] = load_obj(run_labels_path)
    except:
        cprint('Unable to load run_labels!!!!! Initalizing to empty dict')
        global_dict[run_labels] = {}
    if rng == None:
        rng = range(len(global_dict[runs]))
    for j in rng:
        r = global_dict[runs][j]
        if r not in global_dict[run_labels]:
            global_dict[run_labels][r] = blank_labels()
        n = len(gg(opj(global_dict[rgb_1to4_path],r,'*.bag.pkl')))
        labels_str = ""
        ks = sorted(global_dict[run_labels][r])
        labeled = False

        
        if auto_direct_labelling:
            direct_flag = True
            for k in not_direct_modes:
                if k in global_dict[run_labels][r]:

                    if global_dict[run_labels][r][k] != False:
                        direct_flag = False
            if direct_flag:
                global_dict[run_labels][r][direct] = True

        for k in ks:
            if global_dict[run_labels][r][k] != False:
                if k != only_states_1_and_6_good:
                    labeled = True
                labels_str += d2n(i_label_abbreviations[k],':',global_dict[run_labels][r][k],' ')
        if labeled:
            c = 'yellow'
        else:
            c = 'blue'
        cprint(d2n(j,'[',n,'] ',r,'  ',j,'\t',labels_str),c)
        #print global_dict[run_labels][r][direct]
    
def function_set_label(k,v=True):
    """
    function_set_label(k,v)
        function_set_label
    """
    if not global_dict[run_] in global_dict[run_labels]:
        global_dict[run_labels][global_dict[run_]] = {}
    if type(k) != list:
        k = [k]
    for m in k:
        global_dict[run_labels][global_dict[run_]][m] = v
    save_obj(global_dict[run_labels],opj(bair_car_data_path,'run_labels','run_labels_'+time_str()+'.pkl'))

# set the Bag_Folder.pkl for the current run
def function_set_run(j):
    """
    function_set_run()
        function_set_run
    """
    global global_dict
    global_dict[run_] = global_dict[runs][j]
    #cprint(run_ + ' = ' + global_dict[run_])
    Bag_Folder_filename = gg(opj(global_dict[meta_path],global_dict[run_],'Bag_Folder*'))[0]
    B = load_obj(Bag_Folder_filename)
    global_dict[Bag_Folder_Filename] = B
    function_current_run()


def function_set_plot_time_range(t0=-999,t1=-999):
    """
    function_set_plot_time_range
        ST
    """
    r = global_dict[run_]
    B = global_dict[Bag_Folder_Filename]
    ts = np.array(B['data']['raw_timestamps'])
    tsZero = ts - ts[0]
    if t0 < 0:
        t0 = tsZero[0]
        t1 = tsZero[-1]
    figure(r+' stats')
    plt.subplot(5,1,1)
    plt.xlim(t0,t1)
    plt.xlim(t0,t1)
    plt.subplot(5,1,2)
    plt.xlim(t0,t1)
    plt.subplot(5,1,3)
    plt.xlim(t0,t1)
    plt.subplot(5,1,4)
    plt.xlim(t0,t1)


if False: # trying to fix problem
    for i in range(len(global_dict[Bag_Folder_Filename]['data']['state'])):
        if global_dict[Bag_Folder_Filename]['data']['state'][i] == 'no data':
            global_dict[Bag_Folder_Filename]['data']['state'][i] = 0


def function_visualize_run(j=None,do_load_images=True,do_CA=True):
    """
    function_visualize_run()
        function_visualize_run
    """
    if do_CA:
        function_close_all_windows()
    if j != None:
        function_set_run(j)
    global global_dict
    r = global_dict[run_]
    #Bag_Folder_filename = gg(opj(global_dict[meta_path],r,'Bag_Folder*'))[0]
    #B = load_obj(Bag_Folder_filename)
    #global_dict[Bag_Folder_Filename] = B
    B = global_dict[Bag_Folder_Filename]
    L = B['left_image_bound_to_data']
    if global_dict[Bag_Folder_Filename] == None:
        cprint('ERROR, first neet to set run (function_set_run)')
        return
    function_current_run()
    ts = np.array(B['data']['raw_timestamps'])
    tsZero = ts - ts[0]
    dts = B['data']['raw_timestamp_deltas']
    dts_hist = []
    gZero = np.array(B['data']['good_start_timestamps'])
    gZero -= ts[0]

    for j in range(len(dts)):
        dt = dts[j]
        if dt > 0.3:
            dt = 0.3
        dts_hist.append(dt)

    figure(r+' stats',figsize=(7,8))
    clf()
    plt.subplot(5,1,1)
    plt.ylim(-1,8)
    plt.xlim(tsZero[0],tsZero[-1])
    plt.ylabel('state')
    plot(gZero,0.0*array(B['data']['good_start_timestamps']),'gx')
    #plot(tsZero,B['data']['encoder'],'r')  #Sascha: Encoder value not in data.... whyever???
    plot(tsZero,B['data']['state'],'k')
    
    plt.subplot(5,1,2)
    plt.ylim(-5,104)
    plt.xlim(tsZero[0],tsZero[-1])
    plt.ylabel('steer(r) and motor(b)')
    plot(gZero,49+0.0*array(B['data']['good_start_timestamps']),'gx')
    plot(tsZero,B['data']['steer'],'r')
    plot(tsZero,B['data']['motor'],'b')

    plt.subplot(5,1,3)
    plt.xlim(tsZero[0],tsZero[-1])
    plt.ylabel('frame intervals')
    plot(gZero,0.0*array(B['data']['good_start_timestamps']),'gx')
    plot(tsZero,dts)
    plt.ylim(0,0.3)

    plt.subplot(5,1,4)
    plt.xlim(tsZero[0],tsZero[-1])
    plot(gZero,0.0*array(B['data']['good_start_timestamps']),'gx')
    plt.ylabel('state one steps')
    plot(tsZero,array(B['data']['state_one_steps']),'k-')
    #plt.ylim(0,500)

    plt.subplot(5,2,9)
    plt.ylabel('frame intervals')
    bins=plt.hist(dts_hist,bins=100)
    plt.xlim(0,0.3)
    plt.ylim(0,0.001*bins[0].max())
    plt.pause(0.01)

    if do_load_images:
        left_images_ = {}
        right_images_ = {}
        steer_ = {}
        motor_ = {}
        state_ = {}
        bag_paths = sgg(opj(global_dict[rgb_1to4_path],r,'*.bag.pkl'))
        n = len(bag_paths)
        pb = ProgressBar(n)
        j =  0
        cprint('Loading images...')
        for b in bag_paths:
            pb.animate(j); j+=1
            bag_img_dic = load_images(b,color_mode="rgb8",include_flip=False)
            for t in bag_img_dic['left'].keys():
                #print t
                if t in L:
                    steer_[t] = L[t]['steer']
                    motor_[t] = L[t]['motor']
                    state_[t] = L[t]['state']
                    rt = L[t]['right_image']
                    if rt in bag_img_dic['right']:
                        left_images_[t] = bag_img_dic['left'][t]
                        right_images_[t] = bag_img_dic['right'][rt]
                    else:
                        pass
                        #print "rt not in right"
                else:
                    pass
                    #print "t not in left"

        pb.animate(n); print('')
        global_dict[left_images] = left_images_
        global_dict[right_images] = right_images_
        global_dict[steer] = steer_
        global_dict[motor] = motor_
        global_dict[state] = state_
        preview_fig = r+' previews'

        figure(preview_fig)
        clf()

        N = 7
        T0 = B['data']['raw_timestamps'][0]
        Tn1 = B['data']['raw_timestamps'][-1]
        dT = (Tn1-T0)/N**2
        img_title = d2s('total time =',dp((Tn1-T0)/60.0,1),'minutes')
        ctr = 0
        for t in B['data']['raw_timestamps']:
            if t > T0 + ctr * dT:
                if t in left_images_:
                    ctr += 1
                    mi(left_images_[t],preview_fig,[N,N,ctr],do_clf=False)
                    if ctr == N/2:
                        plt.title(img_title)
        pause(0.01)


def function_animate(t0,t1):
    """
    function_animate(t0,t1)
        AR
    """
    function_current_run()
    dT = t1 - t0
    assert(dT>0)
    B = global_dict[Bag_Folder_Filename]
    T0 = t0 + B['data']['raw_timestamps'][0]
    ctr = 0
    s_timer = Timer(1)

    state_one_g_t_zero_dict = {}
    for i in range(len(B['data']['raw_timestamps'])):
        rt = B['data']['raw_timestamps'][i]
        state_one_g_t_zero_dict[rt] = B['data']['state_one_steps'][i]
        #print d2s(rt,state_one_g_t_zero_dict[rt])


    for t in B['data']['raw_timestamps']:
        if t >= T0:
            if s_timer.check():
                print(dp(t-T0+t0,0))
                s_timer.reset()
            if t < T0 + dT:
                rdt = B['data']['raw_timestamp_deltas'][ctr]
                if rdt > 0.1:
                    cprint(d2s('Delay between frames =',rdt),'yellow','on_red')
                    plt.pause(rdt)
                #mi(left_images[t],preview_fig,[N,N,5],do_clf=False)
                #pause(0.0001)
                try:
                    img = global_dict[left_images][t]
                except Exception as e:
                    cprint("********** Exception ***********************",'red')
                    traceback.print_exc(file=sys.stdout)
                    #print(e.message, e.args)
                #print state_one_g_t_zero_dict[t]
                if state_one_g_t_zero_dict[t] < 1:#t not in B['good_timestamps_to_raw_timestamps_indicies__dic']:
                    #img[:,:,0] = img[:,:,1]
                    #img[:,:,2] = img[:,:,1]
                    img[-10:,:,0] = 255
                    img[-10:,:,1:2] = 0
                cv2.imshow('video',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    pass
        ctr += 1



def function_run_loop():
    """
    function_run_loop()
        RL
    """
    print('')
    while True:
        try:
            function_current_run()
            command = raw_input(global_dict[run_] + ' >> ')
            if command in ['q','quit','outta-here!']:
                break
            eval(command)
        except Exception as e:
            cprint("********** Exception ***********************",'red')
            traceback.print_exc(file=sys.stdout)
            #print(e.message, e.args)




def function_save_hdf5(run_num=None,dst_path=opj(bair_car_data_path,'hdf5/runs'),flip=False):
    if run_num != None:
        function_close_all_windows()
        function_set_run(run_num)
        function_visualize_run()
    min_seg_len = 30
    seg_lens = []
    Bag_Folder_Dict_Entry = global_dict[Bag_Folder_Filename]
    Left_Image_Entry = Bag_Folder_Dict_Entry['left_image_bound_to_data']
    state_one_steps=Bag_Folder_Dict_Entry['data']['state_one_steps']

    segment_list = []

    in_segment = False

    for i in range(len(state_one_steps)):
        timestamp = Bag_Folder_Dict_Entry['data']['raw_timestamps'][i]
        if state_one_steps[i] > 0 and timestamp in global_dict[left_images] and timestamp in global_dict[right_images]:
            if not in_segment:
                in_segment = True
                segment_list.append([])
            segment_list[-1].append(timestamp)
        else:
            in_segment = False


    segment_list_with_min_len = []
    for s in segment_list:
        if len(s) >= min_seg_len:
            segment_list_with_min_len.append(s)

    for s in segment_list_with_min_len:
        seg_lens.append(len(s))

    
    if not flip:
        rn = opj(global_dict[run_])
    else:
        rn = opj('flip_'+global_dict[run_])

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    

    hpf5_file = h5py.File(opjD(dst_path,rn+'.hdf5'))
    
    try:
        glabels = hpf5_file.create_group('labels')
    except ValueError as ex:
        print ex
        hpf5_file.close()
        return
    gsegments = hpf5_file.create_group('segments')

    if flip:
        glabels['flip'] = np.array([1])
    else:
        glabels['flip'] = np.array([0])    
    for l in i_labels:
        if l in global_dict[run_labels][global_dict[run_]]:
            if global_dict[run_labels][global_dict[run_]][l]:
                glabels[l] = np.array([1])
            else:
                glabels[l] = np.array([0])
        else:
            glabels[l] = np.array([0])
    for i in range(len(segment_list_with_min_len)):
        segment = segment_list_with_min_len[i]
        left_image_list = []
        right_image_list = []
        steer_list = []
        motor_list = []
        state_list = []
        for j in range(len(segment)):
            timestamp = segment[j]
            limg = global_dict[left_images][timestamp]
            rimg = global_dict[right_images][timestamp]
            st = global_dict[steer][timestamp]
            if flip:
                st -= 49
                st *= -1.0
                st += 49
                left_image_list.append(scipy.fliplr(rimg))
                right_image_list.append(scipy.fliplr(limg))
            else:
                left_image_list.append(limg)
                right_image_list.append(rimg)
            steer_list.append(st)
            motor_list.append(global_dict[motor][timestamp])
            state_list.append(global_dict[state][timestamp])
        gsegments[opj(str(i),'left_timestamp')] = segment
        gsegments[opj(str(i),'left')] = np.array(left_image_list)
        gsegments[opj(str(i),'right')] = np.array(right_image_list)
        gsegments[opj(str(i),'steer')] = np.array(steer_list)
        gsegments[opj(str(i),'motor')] = np.array(motor_list)
        gsegments[opj(str(i),'state')] = np.array(state_list)
    hpf5_file.close()



def mi_or_cv2(img,cv=True,delay=30,title='animate'):
    if cv:
        cv2.imshow(title,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            pass
    else:
        mi(img,title)
        pause(0.0001)



def function_load_hdf5(path):
    F = h5py.File(path)
    Lb = F['labels']
    S = F['segments']
    return Lb,S


def start_at(t):
    while time.time() < t:
        time.sleep(0.1)
        print t-time.time()

def load_animate_hdf5(path,start_at_time=0):
    start_at(start_at_time)
    l,s=function_load_hdf5(path)
    img = False
    for h in range(len(s)):
        if type(img) != bool:
            img *= 0
            img += 128
            mi_or_cv2(img)
        pause(0.5)
        n = str(h)
        for i in range(len(s[n]['left'])):
            img = s[n]['left'][i]
            #print s[n][state][i]
            bar_color = [0,0,0]
            
            if s[n][state][i] == 1:
                bar_color = [0,0,255]
            elif s[n][state][i] == 6:
                bar_color = [255,0,0]
            elif s[n][state][i] == 5:
                bar_color = [255,255,0]
            elif s[n][state][i] == 7:
                bar_color = [255,0,255]
            else:
                bar_color = [0,0,0]
            if i < 2:
                smooth_steer = s[n][steer][i]
            else:
                smooth_steer = (s[n][steer][i] + 0.5*s[n][steer][i-1] + 0.25*s[n][steer][i-2])/1.75
            #print smooth_steer
            apply_rect_to_img(img,smooth_steer,0,99,bar_color,bar_color,0.9,0.1,center=True,reverse=True,horizontal=True)
            apply_rect_to_img(img,s[n][motor][i],0,99,bar_color,bar_color,0.9,0.1,center=True,reverse=True,horizontal=False)
            mi_or_cv2(img)


# filter out left and out in files

def load_hdf5_steer_hist(path,dst_path):
    if len(gg(opj(dst_path,fname(path).replace('hdf5','state_hist_list.pkl')))) == 1:
        print(opj(dst_path,fname(path).replace('hdf5','state_hist_list.pkl'))+' exists')
        return
    try:
        print path
        unix('mkdir -p '+dst_path)
        low_steer = []
        high_steer = []
        labels,segments=function_load_hdf5(path)
        progress_bar = ProgressBar(len(segments))
        state_hist_list = []
        for segment_i in range(len(segments)):
            progress_bar.animate(segment_i)
            state_hist = np.zeros(8)
            n = str(segment_i)
            for i in range(len(segments[n][left])):
                state_hist[int(segments[n][state][i])] += 1
                if i < 2:
                    smooth_steer = segments[n][steer][i]
                else:
                    smooth_steer = (segments[n][steer][i] + 0.5*segments[n][steer][i-1] + 0.25*segments[n][steer][i-2])/1.75
                if smooth_steer < 43 or smooth_steer > 55:
                    high_steer.append([segment_i,i,int(round(smooth_steer))])
                else:
                    low_steer.append([segment_i,i,int(round(smooth_steer))])
            state_hist_list.append(state_hist)
        progress_bar.animate(segment_i)
        assert(len(high_steer)>0)
        assert(len(low_steer)>0)

        save_obj(high_steer,opj(dst_path,fname(path).replace('hdf5','high_steer.pkl')))
        save_obj(low_steer,opj(dst_path,fname(path).replace('hdf5','low_steer.pkl')))
        save_obj(state_hist_list,opj(dst_path,fname(path).replace('hdf5','state_hist_list.pkl')))
    except Exception as e:
        cprint("********** load_hdf5_steer_hist Exception ***********************",'red')
        traceback.print_exc(file=sys.stdout)
        #print(e.message, e.args)


if __name__ == '__main__':
    bair_car_data_path = sys.argv[1]
    HE = function_help    
    SP = function_set_paths
    SP()    
    LR = function_list_runs
    LR()
    ST = function_set_plot_time_range        
    AR = function_animate
    RL = function_run_loop
    A5 = load_animate_hdf5
    #HE()
    #RL()
    
    for i in range(len(global_dict[runs])):
        
        if global_dict[run_labels][global_dict[runs][i]][reject_run] == False:
            print i
            function_save_hdf5(i,flip=False)
            function_save_hdf5(flip=True)
    
    hdf5s = sgg(opj(bair_car_data_path,'hdf5/runs/*.hdf5'))
    ctr = 0
    for h in hdf5s:
        ctr += 1
        print ctr
        load_hdf5_steer_hist(h,opj(bair_car_data_path,'hdf5','segment_metadata'))

    run_codes = {}
    steer_hists = sgg(opj(bair_car_data_path,'hdf5/segment_metadata/*.state_hist_list.pkl'))
    ctr = 0
    combined = []
    for s in steer_hists:
        o = load_obj(s)
        run_codes[ctr] = fname(s).replace('.state_hist_list.pkl','')
        print ctr,run_codes[ctr]
        #for j in range(len(o)):
        #    o[j][3] = ctr
        #    combined.append(o[j])
        ctr += 1
    #save_obj(combined,opjD('combined'))
    save_obj(run_codes,opj(bair_car_data_path,'hdf5/segment_metadata/run_codes'))

    low_steer = []
    high_steer = []
    low_steer_files = sgg(opj(bair_car_data_path,'hdf5/segment_metadata/*.low_steer.pkl'))
    ctr = 0
    for s in low_steer_files:
        print (ctr,s)
        q = load_obj(s)
        for i in range(len(q)):
            q[i].append(ctr)
        low_steer += q
        q = load_obj(s.replace('.low_steer.','.high_steer.'))
        for i in range(len(q)):
            q[i].append(ctr)
        high_steer += q
        ctr += 1
    save_obj(low_steer,opj(bair_car_data_path,'hdf5/segment_metadata/low_steer'))
    save_obj(high_steer,opj(bair_car_data_path,'hdf5/segment_metadata/high_steer'))    