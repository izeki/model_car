#! /usr/bin/python
#//anaconda/bin/python
#
import caffe
#USE_GPU = True
#if USE_GPU:
#	caffe.set_device(0)
#	caffe.set_mode_gpu()
from model_car.utils import *
from model_Car.caffe_net.load_data_into_model_versions import *
import cv2

os.chdir(home_path) # this is for the sake of the train_val.prototxt
os.environ['GLOG_minloglevel'] = '2'

class Caffe_Net:

	def __init__(self,solver_file_path,version,weights_file_mode=None,weights_file_path=None,restore_solver=False):
		self.version = version
		self.solver = setup_solver(solver_file_path)
		self.model_name = solver_file_path.split('/')[-2]
		if restore_solver:
			weights_file_path = most_recent_file_in_folder(weights_file_path,[self.model_name,'solverstate'])
			self.solver.restore(weights_file_path)
			print(d2n("*** self.solver.restore(",weights_file_path,") ***"))
		else:
			if weights_file_mode == 'most recent':
				weights_file_path = most_recent_file_in_folder(weights_file_path,[self.model_name,'caffemodel'])
			elif weights_file_mode == 'this one':
				pass
			elif weights_file_mode == None:
				pass
			else:
				assert(False)
			if weights_file_path != None:
				cprint("loading " + weights_file_path,'red','on_yellow')
				self.solver.net.copy_from(weights_file_path)
		self.train_steps = 0
		self.train_start_time = 0
		self.print_timer = Timer(10)
		self.visualize_timer = Timer(10)
		self.save_loss_timer = Timer(10*60)
		self.loss = []
		self.loss1000 = []
		self.stop_training = False




	def train_step(self,solver=False):

		if solver:
			self.solver.net.blobs['ZED_data_pool2'].data[:] = solver.net.blobs['ZED_data_pool2'].data[:]
			self.solver.net.blobs['metadata'].data[:] = solver.net.blobs['metadata'].data[:]
			self.solver.net.blobs['steer_motor_target_data'].data[:] = solver.net.blobs['steer_motor_target_data'].data[:]
		
		if self.train_steps == 0:
			self.train_start_time = time.time()
		

		self.solver.step(1)

		
		self.train_steps += 1
		a = self.solver.net.blobs['steer_motor_target_data'].data[0,:] - self.solver.net.blobs['ip2'].data[0,:]
		self.loss.append(np.sqrt(a * a).mean())

		if len(self.loss) >= 1000:
			self.loss1000.append(array(self.loss[-1000:]).mean())
			self.loss = []
		if self.print_timer.check():
			print(d2s('self.solver.step(1)',time.time()),self.train_steps, dp(1./((time.time()-self.train_start_time)/(1.*self.train_steps)),2) )
			if len(self.loss1000) > 0:
				print(self.train_steps,self.loss1000[-1])
			print(self.solver.net.blobs['metadata'].data[0,:,5,5])
			cprint(_array_to_int_list(self.solver.net.blobs['steer_motor_target_data'].data[0,:][:]),'green','on_red')
			cprint(_array_to_int_list(self.solver.net.blobs['ip2'].data[0,:][:]),'red','on_green')
			self.print_timer.reset()
		if self.visualize_timer.check():	
			visualize_solver_data(self.solver,self.version,True)
			self.visualize_timer.reset()
		if self.save_loss_timer.check():
			save_obj(self.loss1000,opjD('loss1000'))
			self.save_loss_timer.reset()
		

def print_solver(solver):

	print("")
	
	for l in [(k, v[0].data.shape) for k, v in solver.net.params.items()]:
		print(l)

	print("")
	for l in [(k, v.data.shape) for k, v in solver.net.blobs.items()]:
		if 'split' not in l[0]:
			print(l)



def setup_solver(solver_file_path):
	solver = caffe.SGDSolver(solver_file_path)
	print_solver(solver)
	return solver



"""
def setup_solver(solver_file_path):
	solver = caffe.SGDSolver(solver_file_path)
	for l in [(k, v.data.shape) for k, v in solver.net.blobs.items()]:
		print(l)
	for l in [(k, v[0].data.shape) for k, v in solver.net.params.items()]:
		print(l)
	return solver
"""

def _array_to_int_list(a):
	l = []
	for d in a:
		l.append(int(d*100))
	return l

def load_data_into_model(solver,version,data,flip,show_data,camera_dropout):
	if version == 'version 1':
		return load_data_into_model_version_1(solver,data,flip,show_data,camera_dropout)
	if version == 'version 1b':
		return load_data_into_model_version_1b(solver,data,flip,show_data,camera_dropout)
	if version == 'version 1c':
		return load_data_into_model_version_1c(solver,data,flip,show_data,camera_dropout)
	if version == 'version 2':
		return load_data_into_model_version_2(solver,data,flip,show_data,camera_dropout)
	if version == 'version z3':
		return load_data_into_model_version_z3(solver,data,flip,show_data,camera_dropout)
	assert(False)

def visualize_solver_data(solver,version,flip):
	if version == 'version 1':
		return visualize_solver_data_version_1(solver,flip)
	if version == 'version 1b':
		return visualize_solver_data_version_1b(solver,flip)
	if version == 'version 1c':
		return visualize_solver_data_version_1c(solver,flip)
	if version == 'version 2':
		return visualize_solver_data_version_2(solver,flip)
	if version == 'version z3':
		return visualize_solver_data_version_z3(solver,flip)
	assert(False)


