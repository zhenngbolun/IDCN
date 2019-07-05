import cv2, os, random, math, keras
import numpy as np
from model import *
from keras import optimizers, models, layers
from datetime import datetime
from utils import *
from math import log10
from skimage.measure import compare_ssim

def formatImg(x):
	v_max = np.max(x)
	v_min = np.min(x)
	print(v_max, v_min)
	x = (x-v_min)/(v_max-v_min)
	x = x*255.0+0.5
	x = x.astype(np.uint8)
	return x

def getImg(x):
	x = x*255.0
	x[x>255.0]=255.0
	x[x<0] = 0
	y = x+0.5
	y = y.astype(np.uint8)
	return y

def crop(x,scale):
	h = x.shape[0]
	w = x.shape[1]
	cut = scale
	return x[cut:h-cut,cut:w-cut]

def get_test_list(dataset, QF, sigma = None, is_crop=True):
	print('getting test_list...')
	gt_list = []
	ns_list = []
	name_list = []
	img_dir = None
	count = 0
	if dataset == 'LIVE1':
		img_dir = 'F:/dataset/LIVE1/y/'
		out_dir = 'test_image/LIVE1/'
		tail = 'png'
	if dataset == 'B200':
		img_dir = 'F:\\dataset\\BSDS500\\images\\test\\y\\'
		out_dir = 'test_image/B200/'
		tail = 'jpg'
	if dataset == 'WIN143':
		img_dir = 'F:/dataset/WIN143_resize/'
		out_dir = 'test_image/WIN143/'
		tail = 'png'
	if dataset == 'Classic5':
		img_dir = 'F:\\dataset\\classic5\\'
		out_dir = 'test_image/Classic5/'
		tail = 'bmp'
	if img_dir != None:
		file_list = os.listdir(img_dir)
		for file in file_list:
			s = str.split(file,'.')
			if len(s)!=2 or s[1]!=tail:
				continue
			count += 1
			#print(str(count)+' preprocessing '+s[0]+' ...')
			gt = cv2.imread(img_dir+file, cv2.IMREAD_GRAYSCALE)
			if is_crop:
				gt = crop4interpolation(gt,8)
			cv2.imwrite('test_temp.jpg',gt,[1,QF])
			ns = cv2.imread('test_temp.jpg',cv2.IMREAD_GRAYSCALE)
			gt = gt.astype(np.float32)/255.0
			ns = ns.astype(np.float32)/255.0
			if len(gt.shape) == 2:
				gt = np.expand_dims(gt, 2)
			if len(ns.shape) == 2:
				ns = np.expand_dims(ns, 2)
			#if sigma != None:
			#ns = np.concatenate([ns,sigma[0:ns.shape[0],0:ns.shape[1],:]],axis=-1)
			ns = ns.reshape((1,)+ns.shape)
			gt_list.append(gt)
			ns_list.append(ns)
			name_list.append(s[0])
	return gt_list, ns_list, name_list, out_dir

def getPatch(img, lr_img, size, org_list, lr_list, aug = True):
	x_step = random.randint(size-6, size+14)
	y_step = random.randint(size-6, size+14)
	
	y = img.astype(np.float32)/255.0
	x = lr_img.astype(np.float32)/255.0
	if len(y.shape) == 2:
		y = np.expand_dims(y, 2)
	if len(x.shape) == 2:
		x = np.expand_dims(x, 2)
	for j in range(0,x.shape[1]-size,x_step):
		for i in range(0,x.shape[0]-size,y_step):
			i_end = i+size
			j_end = j+size
			if i_end >= img.shape[0] or j_end >= img.shape[1]:
				continue
			_x = x[i:i_end,j:j_end]
			_y = y[i:i_end,j:j_end]
			if aug:
				method = random.randint(0,5)
				_x = data_augmentation(_x, method)
				_y = data_augmentation(_y, method)
			#d = cv2.dct(x)
			org_list.append(_y)
			lr_list.append(_x)
	return

# Validation or Testing
def validate(model, gt_list, ns_list, out_dir, repeat = False, name_list = None, write=False):
	print("validating... ",datetime.now().strftime('%H:%M:%S'))
	psnr = 0
	count = 0
	l = len(gt_list)
	for i in range(l):
		gt = gt_list[i]
		ns = ns_list[i]	
		dn = model.predict(ns)
		if repeat:
			dn = dn[-1]
		_psnr = 10*log10(1/np.mean((dn[0]-gt)**2))
		psnr += _psnr
		count += 1
		if write:
			print (str(count)+': '+str(_psnr))
			_img = dn[0]
			_img[_img>1] = 1
			_img[_img<0] = 0
			_img = _img*255+0.5
			_img = _img.astype(np.uint8)
			cv2.imwrite(out_dir+name_list[i]+'.bmp',_img)
	print (psnr/count)
	return round(psnr/count,3)

def train(method, QF = 20, pre_batch = 0, aug = True):
	repeat = False
	if method == 'RDN8':
		batch_size = 16
		data_size = 48
		model = RDN_JPEG(8)
	if method == 'MWCNN':
		batch_size = 24
		data_size = 240
		model = MWCNN()
	if method == 'SNet':
		batch_size = 16
		data_size = 48
		model = SNet()
	if method == 'ARCNN':
		batch_size = 128
		data_size = 24
		model = ARCNN()
	if method == 'D_SDNet':
		batch_size = 64
		data_size = 62
		model = D_SDNet()
	if method == 'P_SDNet':
		batch_size = 64
		data_size = 62
		model = P_SDNet()
	if method == 'MemNet':
		repeat = True
		batch_size = 64
		data_size = 31
		model = D_SDNet()

	lr = [1e-4, 5e-5, 2.5e-5, 1e-5, 1e-6]
	lr_index = 0
	batches = 0
	img_count = 0
	best= 0
	fail_count = 0
	gt_list, ns_list, name_list, out_dir = get_test_list('LIVE1',QF)
	model_head = 'weights/'+method+'_Y_QF'+str(QF)+'_'
	if pre_batch > 0:
		model.load_weights(model_head+'best.h5')
		best = validate(model, gt_list,ns_list,out_dir,repeat,name_list,True)
	model.summary()
	print('compiling... lr is ', lr[lr_index])
	model.compile(optimizer=optimizers.Adam(lr = lr[lr_index]), loss='mse')
	org_list = []
	lr_list = []
	
	DIV2K_ROOT_path = 'F:/dataset/DIV2K/'
	images_dir = DIV2K_ROOT_path + 'DIV2K_JPEG/y/'
	lr_dir = DIV2K_ROOT_path+'QF'+str(QF)+'/y/'
	file_list = os.listdir(images_dir)
	file_list = list_filter(file_list, '.png')
	print(len(file_list))
	for k in range(100):
		random.shuffle(file_list)
		for f in file_list:
			s = os.path.splitext(f)
			img = cv2.imread(images_dir+f, cv2.IMREAD_GRAYSCALE)
			lr_img = cv2.imread(lr_dir+s[0]+'.jpg', cv2.IMREAD_GRAYSCALE)
			img_count += 1
			getPatch(img, lr_img, data_size, org_list, lr_list, aug)
			if img_count % 100 == 0:
				#print('proceccing %d to %d, %d', img_count-99, img_count, int(len(org_list)/batch_size))
				org_array = np.array(org_list)
				lr_array = np.array(lr_list)
				indices = list(range(org_array.shape[0]))
				np.random.shuffle(indices)
				for i in range(0,org_array.shape[0]-batch_size, batch_size):
					_y_batch = org_array[indices[i:i+batch_size]]
					_x_batch = lr_array[indices[i:i+batch_size]]
					batches += 1
					if batches <= pre_batch:
						continue
					if repeat:
						y_list = []
						for i in range(7):
							y_list.append(_y_batch)
						train_loss = model.train_on_batch(_x_batch, y_list)
					else:
						train_loss = model.train_on_batch(_x_batch, _y_batch)
					print(batches, train_loss,end='\r')
					if batches % 5000 == 0:
						v1 = validate(model, gt_list,ns_list,out_dir,repeat=repeat)
						if v1>best:
							model.save(model_head+'best.h5')
							fail_count = 0
							best = v1
						else:
							fail_count += 1 
						print('=============', lr[lr_index], batches,v1,best,'=============',datetime.now().strftime('%H:%M:%S'))
						if fail_count >= 4:
							lr_index += 1
							if lr_index < len(lr):
								fail_count = 0
								print('compiling... lr is ', lr[lr_index])
								#model.compile(optimizer=optimizers.SGD(lr = lr, decay=1e-5, momentum=0.9, nesterov=True), loss='mse')
								model.compile(optimizer=optimizers.Adam(lr = lr[lr_index]), loss='mse')
							else:
								return	
				lr_list=[]
				org_list=[]

keras.backend.tensorflow_backend.set_session(get_session())
train('MWCNN', QF=20, pre_batch=0, aug=False)
