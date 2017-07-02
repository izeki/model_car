from model_car.vis import *
from model_car.model.model import *
import h5py

def plot_performance(steer,motor,loss1000):
    figure('loss1000')
    clf()
    plot(loss1000)
    plt.ylim(0.045,0.06)
    plt.title(time_str('Pretty'))
    plt.xlabel(solver_file_path)
    figure('steer')
    clf()

    s1000 = steer[-(min(len(steer),10000)):]
    s = array(s1000)
    plot(s[:,0],s[:,1],'o')
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plot([-1,5,1.5],[-1,5,1.5],'r')
    plt_square()
    plt.title(time_str('Pretty'))
    plt.xlabel(solver_file_path)
    plt.ylabel(dp(np.corrcoef(s[:,0],s[:,1])[0,1],2))

################## Setup Keras ####################################
from keras import optimizers
#solver_file_path = opjh("model_car/net_training/z2_color/solver.prototxt")
version = 'version 1b'
#weights_file_mode = 'most recent' #'this one' #None #'most recent' #'this one'  #None #'most recent'
weights_file_path = opjD('model_car/model_car/model/z2_color_tf.npy') #opjD('z2_color_long_train_21_Jan2017') #None #opjh('kzpy3/caf6/z2_color/z2_color.caffemodel') #None #'/home/karlzipser/Desktop/z2_color' # None #opjD('z2_color')
model = get_model(version, phase='train')
model = load_model_weight(model, weights_file_path)
model.compile(loss = 'mean_squared_error',
                              optimizer = optimizers.SGD(lr = 0.01,  momentum = 0.001, decay = 0.000001, nesterov = True),
                              metrics=['accuracy'])
##############################################################

runs_folder = sys.argv[1]#runs_folder = '/media/karlzipser/ExtraDrive1/caffe_runs'
run_names = sorted(gg(opj(runs_folder,'*.hdf5')),key=natural_keys)
solver_inputs_dic = {}
keys = {}
k_ctr = 0

for hdf5_filename in run_names:
    try:
        solver_inputs_dic[hdf5_filename] = h5py.File(hdf5_filename,'r')
        print hdf5_filename
        kk = solver_inputs_dic[hdf5_filename].keys()
        for k in kk:
            keys[k] = hdf5_filename
            k_ctr += 1
    except Exception as e:
        cprint("********** Exception ***********************",'red')
        print(e.message, e.args)

ks = keys.keys()

cprint(d2s('Using',len(ks),'data entries'),'red','on_yellow')
ctr = 0

loss = []
loss1000 = []

steer = []
motor = []

T = 6
timer = Timer(T)
id_timer = Timer(3*T)

#TODO: Add training iteration
while True: # Training
    random.shuffle(ks)
    for k in ks:
        hdf5_filename = keys[k]
        solver_inputs = solver_inputs_dic[hdf5_filename]
        x_train = {}
        y_train = {}
        x_train['ZED_input'] = solver_inputs[k]['ZED_input']/255.-0.5
        x_train['metadata'] = solver_inputs[k]['metadata']
        y_train['steer_motor_target_data'] = solver_inputs[k]['steer_motor_target_data']
        step_loss = model.train_on_batch({'ZED_data':x_train['ZED_input'], 'metadata':x_train['metadata']}, {'ip2': y_train['steer_motor_target_data']})
        loss.append(step_loss)
        steer.append([y_train['steer_motor_target_data'][0,9],model.layer['ip2'].output[0,9]])
        motor.append([y_train['steer_motor_target_data'][10,19],model.layer['ip2'].output[10,19]])
        if len(loss) >= 1000:
			loss1000.append(array(loss[-1000:]).mean())
			loss = []
        ctr += 1
        if timer.check():
            plot_performance(steer,motor,loss1000)
            timer.reset()
        if id_timer.check():
            cprint(solver_file_path,'blue','on_yellow')
            id_timer.reset()
pass    

"""

figure('loss')
hist(loss)
print np.median(loss)
save_obj(loss,opjD('z2_color/0.056_direct_local_sidewalk_test_data_01Nov16_14h59m31s_Mr_Orange.loss'))

"""

if False: # Testing
    loss = []
    ctr = 0
    for k in ks:
        hdf5_filename = keys[k]
        solver_inputs = solver_inputs_dic[hdf5_filename]
        x_train = {}
        y_train = {}
        x_train['ZED_input'] = solver_inputs[k]['ZED_input']/255.-0.5
        x_train['metadata'] = solver_inputs[k]['metadata']
        y_train['steer_motor_target_data'] = solver_inputs[k]['steer_motor_target_data']
        step_loss = model.train_on_batch({'ZED_data':x_train['ZED_input'], 'metadata':x_train['metadata']}, {'ip2': y_train['steer_motor_target_data']})
        loss.append(step_loss)
        steer.append([y_train['steer_motor_target_data'][0,9],model.layer['ip2'].output[0,9]])
        motor.append([y_train['steer_motor_target_data'][0,19],model.layer['ip2'].output[0,19]])
        if len(loss) >= 1000:
			loss1000.append(array(loss[-1000:]).mean())
			loss = []
        ctr += 1
        if timer.check():
            plot_performance(steer,motor,loss1000)
            timer.reset()
            print ctr
        if id_timer.check():
            cprint(solver_file_path,'blue','on_yellow')
            id_timer.reset()