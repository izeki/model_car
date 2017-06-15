from model_car.vis import *

import cv2
blue = [0,0,0.8]
blue_green = [0,0.5,0.5]

def load_data_into_model_version_z3(solver,data,flip,show_data=False,camera_dropout=False):
	#return True
	if 'left' in data:
		if len(data['left']) >= 60:
			if type(data['left'][0]) == np.ndarray:
				target_data = data['steer'][30:]
				target_data += data['motor'][30:]
				pre_data = data['steer'][:30]
				pre_data += data['motor'][:30]

				
				if not flip:
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['left'][30-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['left'][31-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['right'][30-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['right'][31-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['left'][30-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['left'][31-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['right'][30-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['right'][31-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['left'][30-2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['left'][31-2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['right'][30-2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['right'][31-2][:,:,2]
				

				else: # flip left-right
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['right_flip'][30-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['right_flip'][31-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['left_flip'][30-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['left_flip'][31-2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['right_flip'][30-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['right_flip'][31-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['left_flip'][30-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['left_flip'][31-2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['right_flip'][30-2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['right_flip'][31-2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['left_flip'][30-2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['left_flip'][31-2][:,:,2]

					for i in range(len(target_data)/2):
						t = target_data[i]
						t = t - 49
						t = -t
						t = t + 49
						target_data[i] = t
					for i in range(len(pre_data)/2):
						t = pre_data[i]
						t = t - 49
						t = -t
						t = t + 49
						pre_data[i] = t				



				


				
				Direct = 0.
				Follow = 0.
				Play = 0.
				Furtive = 0.
				Caf = 0.
				Racing = 0

				if 'follow' in data['path']:
					Follow = 1.0
				if 'direct' in data['path']:
					Direct = 1.0
				if 'play' in data['path']:
					Play = 1.0
				if 'furtive' in data['path']:
					Furtive = 1.0
				if 'caffe' in data['path']:
					Caf = 1.0
				if 'racing' in data['path']:
					Racing = 1.0
					Direct = 1.0

				solver.net.blobs['metadata'].data[0,0,:,:] = Racing#target_data[0]/99. #current steer
				solver.net.blobs['metadata'].data[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
				solver.net.blobs['metadata'].data[0,2,:,:] = Follow
				solver.net.blobs['metadata'].data[0,3,:,:] = Direct
				solver.net.blobs['metadata'].data[0,4,:,:] = Play
				solver.net.blobs['metadata'].data[0,5,:,:] = Furtive

				for i in range(len(pre_data)):
					solver.net.blobs['metadata2'].data[0,i,:,:] = pre_data[i]/99.

				for i in range(len(target_data)):
					solver.net.blobs['steer_motor_target_data'].data[0,i] = target_data[i]/99.
				


			if show_data:
				for i in range(len(data['left'])):
					img = data['left'][i].copy()
					steer = data['steer'][i]
					motor = data['motor'][i]
					gyro_x = data['gyro_x'][i]
					gyro_yz_mag = data['gyro_yz_mag'][i]

					apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
					apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

					cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
					if cv2.waitKey(33) & 0xFF == ord('q'):
					    break
		return True

	return False

def visualize_solver_data_version_z3(solver,flip):
	layer_to_use = 'ZED_data_pool2' #'T' #
	r = 2
	for i in range(2):

		data_img_shape = np.shape(solver.net.blobs[layer_to_use].data)
		img = np.zeros((data_img_shape[2],2*data_img_shape[3],3))

		img[:,:data_img_shape[3],0] = solver.net.blobs[layer_to_use].data[0,r+i,:,:].copy()
		img[:,:data_img_shape[3],1] = solver.net.blobs[layer_to_use].data[0,r+i+4,:,:].copy()
		img[:,:data_img_shape[3],2] = solver.net.blobs[layer_to_use].data[0,r+i+8,:,:].copy()
		img[:,data_img_shape[3]:,0] = solver.net.blobs[layer_to_use].data[0,i,:,:].copy()
		img[:,data_img_shape[3]:,1] = solver.net.blobs[layer_to_use].data[0,i+4,:,:].copy()
		img[:,data_img_shape[3]:,2] = solver.net.blobs[layer_to_use].data[0,i+8,:,:].copy()

		img = z2o(img)
		steer = solver.net.blobs['steer_motor_target_data'].data[0,i]*99.
		motor = solver.net.blobs['steer_motor_target_data'].data[0,i+10]*99.

		if flip:
			steer_rect_color = blue_green
		else:
			steer_rect_color = blue
		#apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
		#apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
		figure(layer_to_use,figsize=(4,2))
		mi(img,layer_to_use,img_title=d2s(solver.net.blobs['metadata'].data[0,:,0,0],flip))
		figure('motor steer')
		plt.clf()
		pos = range(0,30)
		neg = np.array(pos) - 30
		plot(pos,0*solver.net.blobs['steer_motor_target_data'].data[0,:30],'k')
		plot(pos,solver.net.blobs['steer_motor_target_data'].data[0,:30]*100-49,'r:')
		plot(neg,solver.net.blobs['metadata2'].data[0,:30,0,0]*100-49,'r:')
		plot(pos,solver.net.blobs['ip2'].data[0,:30]*100-49,'r')
		plot(pos,solver.net.blobs['steer_motor_target_data'].data[0,30:]*100-49,'b:')
		plot(neg,solver.net.blobs['metadata2'].data[0,30:,0,0]*100-49,'b:')
		plot(pos,solver.net.blobs['ip2'].data[0,30:]*100-49,'b')
		plt.ylim(-50,50)
		plt.xlim(-30,30)
		#print solver.net.blobs['steer_motor_target_data'].data[:]
		pause(0.0001)

	return True


def load_data_into_model_version_1(solver,data,flip,show_data=False,camera_dropout=False):
	if 'left' in data:
		if len(data['left']) >= 10:

			if type(data['left'][0]) == np.ndarray:
				target_data = data['steer'][:10]
				target_data += data['motor'][:10]
				#mi(data['left'][0][:,:,:],'left')
				#mi(data['right'][0][:,:,:],'right')
				if not flip:
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['left'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['left'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['right'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['right'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['left'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['left'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['right'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['right'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['left'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['left'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['right'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['right'][1][:,:,2]
					

				else: # flip left-right
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['left_flip'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['left_flip'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['right_flip'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['right_flip'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['left_flip'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['left_flip'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['right_flip'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['right_flip'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['left_flip'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['left_flip'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['right_flip'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['right_flip'][1][:,:,2]

					for i in range(len(target_data)/2):
						t = target_data[i]
						t = t - 49
						t = -t
						t = t + 49
						target_data[i] = t

				#solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] -= 128
				solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] /= 255.0
				solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] -= 0.5

				if False:#camera_dropout:
					ri = random.randint(0,5)
					if ri == 0:
						#print 'here 0'
						#time.sleep(0.1)
						for e in [0,1,4,5,8,9]:
							solver.net.blobs['ZED_data_pool2'].data[0,e,:,:] *= 0
					elif ri == 1:
						#print 'here 1'
						#time.sleep(0.1)
						for e in [2,3,6,7,10,11]:
							solver.net.blobs['ZED_data_pool2'].data[0,2:3,:,:] *= 0
					else:
						pass


				Direct = 0.
				Follow = 0.
				Play = 0.
				Furtive = 0.
				Caf = 0.
				Racing = 0

				if 'follow' in data['path']:
					Follow = 1.0
				if 'direct' in data['path']:
					Direct = 1.0
				if 'play' in data['path']:
					Play = 1.0
				if 'furtive' in data['path']:
					Furtive = 1.0
				if 'caffe' in data['path']:
					Caf = 1.0
				if 'racing' in data['path']:
					Racing = 1.0
					Direct = 1.0

				solver.net.blobs['metadata'].data[0,0,:,:] = Racing#target_data[0]/99. #current steer
				solver.net.blobs['metadata'].data[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
				solver.net.blobs['metadata'].data[0,2,:,:] = Follow
				solver.net.blobs['metadata'].data[0,3,:,:] = Direct
				solver.net.blobs['metadata'].data[0,4,:,:] = Play
				solver.net.blobs['metadata'].data[0,5,:,:] = Furtive

				for i in range(len(target_data)):
					solver.net.blobs['steer_motor_target_data'].data[0,i] = target_data[i]/99.



			if show_data:
				for i in range(len(data['left'])):
					img = data['left'][i].copy()
					steer = data['steer'][i]
					motor = data['motor'][i]
					gyro_x = data['gyro_x'][i]
					gyro_yz_mag = data['gyro_yz_mag'][i]

					apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
					apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

					cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
					if cv2.waitKey(33) & 0xFF == ord('q'):
					    break
		return True

	return False

def visualize_solver_data_version_1(solver,flip):
	layer_to_use = 'ZED_data_pool2'
	for i in range(2):
		#data_img_shape = np.shape(solver.net.blobs['ZED_data_pool2'].data)
		data_img_shape = np.shape(solver.net.blobs[layer_to_use].data)
		img = np.zeros((data_img_shape[2],data_img_shape[3],3))

		#img[:,:,0] = z2o(solver.net.blobs['ZED_data_pool2'].data[0,i,:,:])
		#img[:,:,1] = img[:,:,0].copy()
		#img[:,:,2] = img[:,:,0].copy()

		#img[:,:,0] = z2o(solver.net.blobs['ZED_data_pool2'].data[0,i,:,:])
		#img[:,:,1] = z2o(solver.net.blobs['ZED_data_pool2'].data[0,i+4,:,:])
		#img[:,:,2] = z2o(solver.net.blobs['ZED_data_pool2'].data[0,i+8,:,:])

		img[:,:,0] = solver.net.blobs[layer_to_use].data[0,i,:,:].copy()
		img[:,:,1] = solver.net.blobs[layer_to_use].data[0,i+4,:,:].copy()
		img[:,:,2] = solver.net.blobs[layer_to_use].data[0,i+8,:,:].copy()


		img = z2o(img)
		#mi(img,'img')
		steer = solver.net.blobs['steer_motor_target_data'].data[0,i]*99.
		motor = solver.net.blobs['steer_motor_target_data'].data[0,i+10]*99.

		if flip:
			steer_rect_color = blue_green
		else:
			steer_rect_color = blue
		#print(img.min(),img.max())
		apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
		apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
		mi(img,layer_to_use,img_title=d2s(solver.net.blobs['metadata'].data[0,:,0,0]))

		pause(0.001)
		#cv2.imshow('left',cv2.cvtColor((255*img).astype(np.uint8),cv2.COLOR_RGB2BGR))#.astype('uint8')
		#if cv2.waitKey(33) & 0xFF == ord('q'):
		#    break
	return True



def load_data_into_model_version_1b(solver,data,flip,show_data=False,camera_dropout=False):
	#return True
	if 'left' in data:
		if len(data['left']) >= 10:

			if type(data['left'][0]) == np.ndarray:
				target_data = data['steer'][:10]
				target_data += data['motor'][:10]
				#mi(data['left'][0][:,:,:],'left')
				#mi(data['right'][0][:,:,:],'right')
				
				if not flip:
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['left'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['left'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['right'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['right'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['left'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['left'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['right'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['right'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['left'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['left'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['right'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['right'][1][:,:,2]
					

				else: # flip left-right
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['right_flip'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['right_flip'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['left_flip'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['left_flip'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['right_flip'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['right_flip'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['left_flip'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['left_flip'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['right_flip'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['right_flip'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['left_flip'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['left_flip'][1][:,:,2]

					for i in range(len(target_data)/2):
						t = target_data[i]
						t = t - 49
						t = -t
						t = t + 49
						target_data[i] = t
				
				#print type(solver.net.blobs['ZED_data_pool2'].data[0,0,0,0])
				if False:
					solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] /= 255.0
					solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] -= 0.5
				


				
				Direct = 0.
				Follow = 0.
				Play = 0.
				Furtive = 0.
				Caf = 0.
				Racing = 0

				if 'follow' in data['path']:
					Follow = 1.0
				if 'direct' in data['path']:
					Direct = 1.0
				if 'play' in data['path']:
					Play = 1.0
				if 'furtive' in data['path']:
					Furtive = 1.0
				if 'caffe' in data['path']:
					Caf = 1.0
				if 'racing' in data['path']:
					Racing = 1.0
					Direct = 1.0

				solver.net.blobs['metadata'].data[0,0,:,:] = Racing#target_data[0]/99. #current steer
				solver.net.blobs['metadata'].data[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
				solver.net.blobs['metadata'].data[0,2,:,:] = Follow
				solver.net.blobs['metadata'].data[0,3,:,:] = Direct
				solver.net.blobs['metadata'].data[0,4,:,:] = Play
				solver.net.blobs['metadata'].data[0,5,:,:] = Furtive

				for i in range(len(target_data)):
					solver.net.blobs['steer_motor_target_data'].data[0,i] = target_data[i]/99.
				


			if show_data:
				for i in range(len(data['left'])):
					img = data['left'][i].copy()
					steer = data['steer'][i]
					motor = data['motor'][i]
					gyro_x = data['gyro_x'][i]
					gyro_yz_mag = data['gyro_yz_mag'][i]

					apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
					apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

					cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
					if cv2.waitKey(33) & 0xFF == ord('q'):
					    break
		return True

	return False

def visualize_solver_data_version_1b(solver,flip):
	layer_to_use = 'ZED_data_pool2' #'T' #
	r = 2
	for i in range(2):

		data_img_shape = np.shape(solver.net.blobs[layer_to_use].data)
		img = np.zeros((data_img_shape[2],2*data_img_shape[3],3))

		img[:,:data_img_shape[3],0] = solver.net.blobs[layer_to_use].data[0,r+i,:,:].copy()
		img[:,:data_img_shape[3],1] = solver.net.blobs[layer_to_use].data[0,r+i+4,:,:].copy()
		img[:,:data_img_shape[3],2] = solver.net.blobs[layer_to_use].data[0,r+i+8,:,:].copy()
		img[:,data_img_shape[3]:,0] = solver.net.blobs[layer_to_use].data[0,i,:,:].copy()
		img[:,data_img_shape[3]:,1] = solver.net.blobs[layer_to_use].data[0,i+4,:,:].copy()
		img[:,data_img_shape[3]:,2] = solver.net.blobs[layer_to_use].data[0,i+8,:,:].copy()

		img = z2o(img)
		steer = solver.net.blobs['steer_motor_target_data'].data[0,i]*99.
		motor = solver.net.blobs['steer_motor_target_data'].data[0,i+10]*99.

		if flip:
			steer_rect_color = blue_green
		else:
			steer_rect_color = blue
		#apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
		#apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
		figure(layer_to_use,figsize=(4,2))
		mi(img,layer_to_use,img_title=d2s(solver.net.blobs['metadata'].data[0,:,0,0],flip))
		#print solver.net.blobs['steer_motor_target_data'].data[:]
		pause(0.0001)

	return True


def load_data_into_model_version_2(solver,data,flip,show_data=False,camera_dropout=False):
	if 'left' in data:
		if len(data['left']) >= 10:

			if type(data['left'][0]) == np.ndarray:
				steer0 = data['steer'][0]
				motor0 = data['motor'][0]
				target_data = list(array(data['steer'][:]) - steer0)
				target_data += list(array(data['motor'][:]) - motor0)
				target_data += list(array(data['gyro_yz_mag'][:])/10.)

				if not flip:
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['left'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['left'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['left'][2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['right'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['right'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['right'][2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['left'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['left'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['left'][2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['right'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['right'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['right'][2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,12,:,:] = data['left'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,13,:,:] = data['left'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,14,:,:] = data['left'][2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,15,:,:] = data['right'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,16,:,:] = data['right'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,17,:,:] = data['right'][2][:,:,2]
					

				else: # flip left-right
					solver.net.blobs['ZED_data_pool2'].data[0,0,:,:] = data['right_flip'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,1,:,:] = data['right_flip'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,2,:,:] = data['right_flip'][2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,3,:,:] = data['left_flip'][0][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,4,:,:] = data['left_flip'][1][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,5,:,:] = data['left_flip'][2][:,:,0]
					solver.net.blobs['ZED_data_pool2'].data[0,6,:,:] = data['right_flip'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,7,:,:] = data['right_flip'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,8,:,:] = data['right_flip'][2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,9,:,:] = data['left_flip'][0][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,10,:,:] = data['left_flip'][1][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,11,:,:] = data['left_flip'][2][:,:,1]
					solver.net.blobs['ZED_data_pool2'].data[0,12,:,:] = data['right_flip'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,13,:,:] = data['right_flip'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,14,:,:] = data['right_flip'][2][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,15,:,:] = data['left_flip'][0][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,16,:,:] = data['left_flip'][1][:,:,2]
					solver.net.blobs['ZED_data_pool2'].data[0,17,:,:] = data['left_flip'][2][:,:,2]

					for i in range(30):
						t = target_data[i]
						#t = t - 49
						t = -t
						#t = t + 49
						target_data[i] = t

				solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] /= 255.0
				solver.net.blobs['ZED_data_pool2'].data[:,:,:,:] -= 0.5

				Direct = 0.
				Follow = 0.
				Play = 0.
				Furtive = 0.
				Caf = 0.
				Racing = 0

				if 'follow' in data['path']:
					Follow = 1.0
				if 'direct' in data['path']:
					Direct = 1.0
				if 'play' in data['path']:
					Play = 1.0
				if 'furtive' in data['path']:
					Furtive = 1.0
				if 'caffe' in data['path']:
					Caf = 1.0
				if 'racing' in data['path']:
					Racing = 1.0
					Direct = 1.0

				solver.net.blobs['metadata'].data[0,0,:,:] = Racing#target_data[0]/99. #current steer
				solver.net.blobs['metadata'].data[0,1,:,:] = Caf#target_data[len(target_data)/2]/99. #current motor
				solver.net.blobs['metadata'].data[0,2,:,:] = Follow
				solver.net.blobs['metadata'].data[0,3,:,:] = Direct
				solver.net.blobs['metadata'].data[0,4,:,:] = Play
				solver.net.blobs['metadata'].data[0,5,:,:] = Furtive
				solver.net.blobs['metadata'].data[0,6,:,:] = steer0/99.
				solver.net.blobs['metadata'].data[0,7,:,:] = motor0/99.

				for i in range(len(target_data)):
					solver.net.blobs['steer_motor_target_data'].data[0,i] = target_data[i]/99.



			if show_data:
				for i in range(len(data['left'])):
					img = data['left'][i].copy()
					steer = data['steer'][i]
					motor = data['motor'][i]
					gyro_x = data['gyro_x'][i]
					gyro_yz_mag = data['gyro_yz_mag'][i]

					apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
					apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_yz_mag,-150,150,steer_rect_color,steer_rect_color,0.13,0.03,center=True,reverse=True,horizontal=False)
					apply_rect_to_img(img,gyro_x,-150,150,steer_rect_color,steer_rect_color,0.16,0.03,center=True,reverse=True,horizontal=False)

					cv2.imshow('left',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))#.astype('uint8')
					if cv2.waitKey(33) & 0xFF == ord('q'):
					    break
		return True

	return False

def visualize_solver_data_version_2(solver,flip):		
	layer_to_use = 'ZED_data_pool2'
	figure(layer_to_use+'_')
	clf()
	plot(solver.net.blobs['steer_motor_target_data'].data[0,:30],'ro-')
	plot(solver.net.blobs['steer_motor_target_data'].data[0,30:60],'rx-')
	plot(solver.net.blobs['steer_motor_target_data'].data[0,60:],'r')
	plot(solver.net.blobs['ip2'].data[0,:30],'bo-')
	plot(solver.net.blobs['ip2'].data[0,30:60],'bx-')
	plot(solver.net.blobs['ip2'].data[0,60:],'b')
	r = 3
	for i in range(3):

		data_img_shape = np.shape(solver.net.blobs[layer_to_use].data)
		img = np.zeros((data_img_shape[2],2*data_img_shape[3],3))

		img[:,:data_img_shape[3],0] = solver.net.blobs[layer_to_use].data[0,r+i,:,:].copy()
		img[:,:data_img_shape[3],1] = solver.net.blobs[layer_to_use].data[0,r+i+6,:,:].copy()
		img[:,:data_img_shape[3],2] = solver.net.blobs[layer_to_use].data[0,r+i+12,:,:].copy()
		img[:,data_img_shape[3]:,0] = solver.net.blobs[layer_to_use].data[0,i,:,:].copy()
		img[:,data_img_shape[3]:,1] = solver.net.blobs[layer_to_use].data[0,i+6,:,:].copy()
		img[:,data_img_shape[3]:,2] = solver.net.blobs[layer_to_use].data[0,i+12,:,:].copy()

		img = z2o(img)
		steer = solver.net.blobs['steer_motor_target_data'].data[0,i]*99.
		motor = solver.net.blobs['steer_motor_target_data'].data[0,i+10]*99.

		if flip:
			steer_rect_color = blue_green
		else:
			steer_rect_color = blue
		#apply_rect_to_img(img,steer,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=True)
		#apply_rect_to_img(img,motor,0,99,steer_rect_color,steer_rect_color,0.9,0.1,center=True,reverse=True,horizontal=False)
		mi(img,layer_to_use,img_title=d2s(solver.net.blobs['metadata'].data[0,:,0,0],flip))


		pause(0.001)

	return True


