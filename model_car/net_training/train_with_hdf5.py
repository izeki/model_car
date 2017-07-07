from model_car.vis import *
from model_car.model.model import *
import model_car.net_training.type_handlers.get_data_with_hdf5 as get_data_with_hdf5

DISPLAY = True

ignore = ['reject_run','left','out1_in2','Smyth','racing'] # runs with these labels are ignored
require_one = [] # at least one of this type of run lable is required
use_states = [1]
rate_timer_interval = 5.
print_timer = Timer(5)



################## Setup Keras ####################################
from keras import optimizers
MODEL = 'z2_color'
version = 'version 1b'
bair_car_data_path = opjD(sys.argv[1]) # '/media/karlzipser/ExtraDrive4/bair_car_data_new_28April2017'#opjD('bair_car_data_new')
#weights_file_path =  most_recent_file_in_folder(opjD(fname(opjh(REPO,CAF,MODEL))))
weights_file_path = opjD(sys.argv[2]) #weights_file_path = opjD('model_car/model_car/model/z2_color_tf.npy') 
N_FRAMES = 2 # how many timesteps with images.
N_STEPS = 10 # how many timestamps with non-image data
gpu = 0
model = get_model(version, phase='train')    
if weights_file_path:
    print(d2s("Copying weights from",weights_file_path)
    model = load_model_weight(model, weights_file_path)
else:
    print("No weights loaded to")
model.compile(loss = 'mean_squared_error',
                              optimizer = optimizers.SGD(lr = 0.01,  momentum = 0.001, decay = 0.000001, nesterov = True),
                              metrics=['accuracy'])          
time.sleep(5)              
##############################################################    


hdf5_runs_path = opj(bair_car_data_path,'hdf5/runs')
hdf5_segment_metadata_path = opj(bair_car_data_path,'hdf5/segment_metadata')


loss10000 = []
loss = []

rate_timer = Timer(rate_timer_interval)
rate_ctr = 0

get_data_with_hdf5.load_Segment_Data(hdf5_segment_metadata_path,hdf5_runs_path)



print('\nloading low_steer... (takes awhile)')
low_steer = load_obj(opj(hdf5_segment_metadata_path,'low_steer'))
print('\nloading high_steer... (takes awhile)')
high_steer = load_obj(opj(hdf5_segment_metadata_path,'high_steer'))
len_high_steer = len(high_steer)
len_low_steer = len(low_steer)

ctr_low = -1 # These counter keep track of position in segment lists, and when to reshuffle.
ctr_high = -1


def get_data_considering_high_low_steer():
    global ctr_low
    global ctr_high
    global low_steer
    global high_steer

    if ctr_low >= len_low_steer:
        ctr_low = -1
    if ctr_high >= len_high_steer:
        ctr_high = -1
    if ctr_low == -1:
        random.shuffle(low_steer) # shuffle data before using (again)
        ctr_low = 0
    if ctr_high == -1:
        random.shuffle(high_steer)
        ctr_high = 0
        
    if random.random() < 0.5: # len_high_steer/(len_low_steer+len_high_steer+0.0): # with some probability choose a low_steer element
        choice = low_steer[ctr_low]
        ctr_low += 1
    else:
        choice = high_steer[ctr_high]
        ctr_high += 1
    run_code = choice[3]
    seg_num = choice[0]
    offset = choice[1]

    data = get_data_with_hdf5.get_data(run_code,seg_num,offset,N_STEPS,offset+0,N_FRAMES,ignore=ignore,require_one=require_one,use_states=use_states)

    return data



def array_to_int_list(a):
    l = []
    for d in a:
        l.append(int(d*100))
    return l



if DISPLAY:
    figure('steer',figsize=(3,2))
    figure('loss',figsize=(3,2))
    figure('high low steer histograms',figsize=(2,1))
    clf()
    plt.hist(array(low_steer)[:,2],bins=range(0,100))
    plt.hist(array(high_steer)[:,2],bins=range(0,100))
    figure(1)

while True:    
    data = get_data_considering_high_low_steer()    
    # put data into model
    input_height = data['left'][0].shape[0]
    input_width =  data['left'][0].shape[1]          
    ZED_input = np.zeros((1, 12, input_height, input_width))
    ctr = 0
	for c in range(3):
		for camera in ['left','right']:
			for t in range(2):
				ZED_input[0,ctr,:,:] = data[camera][t][:,:,c]
				ctr += 1
    meta_input = np.zeros((1,6, 14, 26))
	Racing = 0
	AI = 0
	Follow = 0
	Direct = 0
	Play = 0
	Furtive = 0
	if data['labels']['racing']:
		Racing = 1.0
	if data['states'][0] == 6:
		AI = 1.0
	if data['labels']['follow']:
		Follow = 1.0
	if data['labels']['direct']:
		Direct = 1.0
	if data['labels']['play']:
		Play = 1.0
	if data['labels']['furtive']:
		Furtive = 1.0
    meta_input[0,0,:,:]= Racing
    meta_input[0,1,:,:]= AI
    meta_input[0,2,:,:]= Follow
    meta_input[0,3,:,:]= Direct
    meta_input[0,4,:,:]= Play
    meta_input[0,5,:,:]= Furtive
    steer_motor_target_data = np.zeros((1,20))
    steer_motor_target_data[0][0:10] = data['steer'][-10:]/99.
    steer_motor_target_data[0][10:] = data['motor'][-10:]/99.
    step_loss = model.train_on_batch({'ZED_input': ZED_input, 'meta_input':meta_input}, {'ip2': steer_motor_target_data})
    if not DISPLAY:
        if print_timer.check():
            print(meta_input[0,:,14,26])
            print(array_to_int_list(steer_motor_target_data[0,:]))
            print(array_to_int_list(model.layer['ip2'].output[0,:]))
            print_timer.reset()

    if DISPLAY:
        # The training step. Everything below is for display.
        rate_ctr += 1
        if rate_timer.check():
            print(d2s('rate =',dp(rate_ctr/rate_timer_interval,2),'Hz'))
            rate_timer.reset()
            rate_ctr = 0
        loss.append(step_loss)
        if len(loss) >= 10000:
            loss10000.append(array(loss[-10000:]).mean())
            loss = []
            figure('loss');clf()
            lm = min(len(loss10000),100)
            plot(loss10000[-lm:])
            print(d2s('loss10000 =',loss10000[-1]))
        if print_timer.check():

            print(meta_input[0,:,14,26])

            cprint(array_to_int_list(steer_motor_target_data[0,:]),'green','on_red')
            cprint(array_to_int_list(model.layer['ip2'].output[0,:]),'red','on_green')
            
            figure('steer')
            clf()
            
            t = steer_motor_target_data[0,:]
            o = model.layer['ip2'].output[0,:]
            ylim(-0.05,1.05);xlim(0,len(t))
            plot([-1,60],[0.49,0.49],'k');plot(o,'og'); plot(t,'or'); plt.title(data['name'])
            
            #print(shape(Solver.solver.net.blobs['steer_motor_target_data'].data))
            #print Solver.solver.net.blobs['steer_motor_target_data'].data[-1,:]
            #print Solver.solver.net.blobs['ip2'].data[-1,:]

            mi_or_cv2_animate(data['left'],delay=33)
            pause(0.001)
            print_timer.reset()





