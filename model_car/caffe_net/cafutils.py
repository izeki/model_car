from model_car.vis import *
import h5py
import caffe


def write_solver(
	model_path='kzpy3/caf6/z2_color_streamline',
	test_iter=1,
	test_interval=1000000,
	test_initialization='false',
	base_lr = 0.01,
	momentum=0.0001,
	weight_decay='0.000005',
	lr_policy="inv",
	gamma=0.0001,
	power=0.75,
	display=20000,
	max_iter=10000000,
	snapshot=100000
	):

	solver_str = """
net: \""""+opj(model_path,"train_val.prototxt")+"""\"
test_iter: """+d2n(test_iter)+"""
test_interval: """+d2n(test_interval)+"""
test_initialization: """+d2n(test_initialization)+"""
base_lr: """+d2n(base_lr)+"""  
momentum: """+d2n(momentum)+"""
weight_decay: """+d2n(weight_decay)+"""
lr_policy: \""""+d2n(lr_policy)+"""\"
gamma: """+d2n(gamma)+"""
power: """+d2n(power)+"""
display: """+d2n(display)+"""
max_iter: """+d2n(max_iter)+"""
snapshot: """+d2n(snapshot)+"""
snapshot_prefix: \""""+opjD(fname(model_path),fname(model_path))+"""\"
	"""
	print solver_str
	list_of_strings_to_txt_file(opj(model_path,'solver.prototxt'),[solver_str])
	




def plot_performance(steer,motor,loss1000,solver_file_path,ylims=None):
	figure(solver_file_path + ' loss1000',figsize=(2,4))
	clf()
	plot(loss1000[-min(len(loss1000),1000):])
	if ylims != None:
		plt.ylim(ylims[0],ylims[1])
	plt.title(time_str('Pretty'))
	plt.xlabel(solver_file_path)
	figure(solver_file_path + ' steer',figsize=(4,2))
	clf()
	s1000 = steer[-(min(len(steer),10000)):]
	s = array(s1000)
	m1000 = motor[-(min(len(motor),10000)):]
	m = array(m1000)
	plt.subplot(1,2,1)
	plot(s[:1000,0],s[:1000,1],'o')
	plt.xlim(0,1.0)
	plt.ylim(0,1.0)
	plot([-1,5,1.5],[-1,5,1.5],'r')
	plt_square()
	plt.xlabel(solver_file_path)
	plt.ylabel(dp(np.corrcoef(s[:,0],s[:,1])[0,1],2))
	plt.subplot(1,2,2)
	plot(m[:1000,0],m[:1000,1],'o')
	plt.xlim(0,1.0)
	plt.ylim(0,1.0)
	plot([-1,5,1.5],[-1,5,1.5],'r')
	plt_square()	
	plt.title(time_str('Pretty'))
	plt.xlabel(solver_file_path)
	plt.ylabel(dp(np.corrcoef(m[:,0],m[:,1])[0,1],2))



def get_solver_inputs_dic_ks(runs_folder,to_require=[''],to_ignore=[]):
	assert(len(gg(opj(runs_folder,'*'))) > 0)
	run_names = sorted(gg(opj(runs_folder,'*.hdf5')),key=natural_keys)
	solver_inputs_dic = {}
	keys = {}
	k_ctr = 0
	for hdf5_filename in run_names:
		if (str_contains_one(hdf5_filename,to_ignore)) or (not str_contains_one(hdf5_filename,to_require)):
			continue
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

	return solver_inputs_dic,keys


def train(caffe_net,solver_inputs_dic,keys,version,model_file_path,time_limit=None,ylims=None):
	if time_limit != None:
		limit_timer = Timer(time_limit)
	ctr = 0
	steer = []
	motor = []
	T = 6
	timer = Timer(T)
	id_timer = Timer(3*T)
	while True:
		ks = keys.keys()
		random.shuffle(ks)
		for k in ks:
			if time_limit != None:
				if limit_timer.check():
					return
			hdf5_filename = keys[k]
			solver_inputs = solver_inputs_dic[hdf5_filename]
			caffe_net.solver.net.blobs['ZED_data_pool2'].data[:] = solver_inputs[k]['ZED_data_pool2'][:]/255.-0.5
			caffe_net.solver.net.blobs['metadata'].data[:] = solver_inputs[k]['metadata'][:]
			#caffe_net.solver.net.blobs['metadata2'].data[:] = solver_inputs[k]['metadata2'][:]
			caffe_net.solver.net.blobs['steer_motor_target_data'].data[:] = solver_inputs[k]['steer_motor_target_data'][:]
			caffe_net.train_step()
			steer.append([caffe_net.solver.net.blobs['steer_motor_target_data'].data[0,9],caffe_net.solver.net.blobs['ip2'].data[0,9]])
			motor.append([caffe_net.solver.net.blobs['steer_motor_target_data'].data[0,39],caffe_net.solver.net.blobs['ip2'].data[0,19]])
			ctr += 1
			if timer.check():
				plot_performance(steer,motor,caffe_net.loss1000,"TRAIN " + model_file_path,ylims)
				timer.reset()
			if id_timer.check():
				cprint(model_file_path,'blue','on_yellow')
				id_timer.reset()


def test(caffe_net,solver_inputs_dic,keys,version,model_path,time_limit):
	if time_limit != None:
		limit_timer = Timer(time_limit)
	ctr = 0
	steer = []
	motor = []
	T = 6
	timer = Timer(T)
	id_timer = Timer(3*T)
	ks = keys.keys()
	random.shuffle(ks) # shuffling lets us use subsamples of the test data
	loss = []
	for k in ks:
		if time_limit != None:
			if limit_timer.check():
				break
		hdf5_filename = keys[k]
		solver_inputs = solver_inputs_dic[hdf5_filename]
		caffe_net.solver.net.blobs['ZED_data_pool2'].data[:] = solver_inputs[k]['ZED_data_pool2'][:]/255.-0.5
		caffe_net.solver.net.blobs['metadata'].data[:] = solver_inputs[k]['metadata'][:]
		caffe_net.solver.net.blobs['steer_motor_target_data'].data[:] = solver_inputs[k]['steer_motor_target_data'][:]
		caffe_net.train_step()
		a = caffe_net.solver.net.blobs['steer_motor_target_data'].data[0,:] - caffe_net.solver.net.blobs['ip2'].data[0,:]
		loss.append(np.sqrt(a * a).mean())
		steer.append([caffe_net.solver.net.blobs['steer_motor_target_data'].data[0,9],caffe_net.solver.net.blobs['ip2'].data[0,9]])
		motor.append([caffe_net.solver.net.blobs['steer_motor_target_data'].data[0,19],caffe_net.solver.net.blobs['ip2'].data[0,19]])
		ctr += 1
		if timer.check():
			plot_performance(steer,motor,caffe_net.loss1000,"TEST " + model_path)
			timer.reset()
			print ctr
		if id_timer.check():
			cprint(model_path,'blue','on_yellow')
			id_timer.reset()
	figure('TEST loss',figsize=(2,2))
	hist(loss)
	median_loss = np.median(loss)
	plt.title(d2s("TEST",model_path,'median loss =',median_loss))
	print median_loss
	return median_loss

	

