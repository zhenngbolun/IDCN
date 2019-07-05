import tensorflow as tf
import os, cv2, random, keras
import numpy as np
from model import *
from keras import optimizers
from keras.utils import multi_gpu_model
from math import log10
from utils import *
from datetime import datetime
import time

# generate gt_list and ns_list for testing and validation
def get_test_list(dataset, QF, sigma = None, is_crop=False):
	print('getting test_list...')
	gt_list = []
	ns_list = []
	name_list = []
	img_dir = None
	count = 0
	if dataset == 'LIVE1':
		img_dir = 'F:/dataset/LIVE1/'
		out_dir = 'test_image/LIVE1/'
		tail = 'bmp'
	if dataset == 'B200':
		img_dir = 'F:\\dataset\\BSDS500\\images\\test\\'
		out_dir = 'test_image/B200/'
		tail = 'jpg'
	if dataset == 'WIN143':
		img_dir = 'F:/dataset/WIN143_resize/'
		out_dir = 'test_image/WIN143/'
		tail = 'png'

	if img_dir != None:
		file_list = os.listdir(img_dir)
		for file in file_list:
			s = str.split(file,'.')
			if len(s)!=2 or s[1]!=tail:
				continue
			count += 1
			#print(str(count)+' preprocessing '+s[0]+' ...')
			gt = cv2.imread(img_dir+file)
			if is_crop:
				gt = crop(gt,8)
			cv2.imwrite('test_temp.jpg',gt,[1,QF])
			ns = cv2.imread('test_temp.jpg')
			gt = gt.astype(np.float32)/255.0
			ns = ns.astype(np.float32)/255.0
			#if sigma != None:
			ns = np.concatenate([ns,sigma[0:ns.shape[0],0:ns.shape[1],:]],axis=-1)
			ns = ns.reshape((1,)+ns.shape)
			gt_list.append(gt)
			ns_list.append(ns)
			name_list.append(s[0])
	return gt_list, ns_list, name_list, out_dir

# Validation or Testing
def validate(model, gt_list, ns_list, out_dir, name_list = None, write=False):
	print("validating... ",datetime.now().strftime('%H:%M:%S'))
	psnr = 0
	count = 0
	l = len(gt_list)
	for i in range(l):
		gt = gt_list[i]
		ns = ns_list[i]	
		dn = model.predict(ns)
		_psnr = 10*log10(1/np.mean(((dn[0])-(gt))**2))
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

#Training
multi_gpu = False
QF = 10
if multi_gpu:	
	print ('use multi gpu mode!')
	os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
keras.backend.tensorflow_backend.set_session(get_session())

#DIV2K_ROOT_path = '/media/scs4450/hard/zbl/srcnn/src/train/DIV2K/'
images_dir = 'F:/dataset/DIV2K/DIV2K_JPEG/'
LR_dir = 'F:/dataset/DIV2K/QF'+str(QF)+'/'

qy = get_table(luminance_quant_table, QF)
qc = get_table(chrominance_quant_table, QF)
sigma = get_sigma_c1(QF)



evaluation = 'psnr'
datas_list = []
x_list = []
y_list = []
pre_batches = 0
min_loss = 0
fail_count = 0
data_size = 43
batch_size = 16
lr = [1e-4, 5e-5, 2.5e-5, 1e-5, 1e-6]
lr_index = 0
compile_flag = False
PyramidCells = ((3,2,1,1,1,1),)
model = IDCN(64, 8, PyramidCells, 64, qy,qc)

gt_list, ns_list, name_list, out_dir = get_test_list('LIVE1',QF, sigma=sigma, is_crop=False)
model_head = 'weights/IDCN_c1_'+str(QF)+'_8_64_'
temp_path = 'temp_v'+str(random.randint(100000,999999))+'.jpg'

#model.load_weights('weights/IDCN_c1_10_8_64_best.h5', by_name=True)
#min_loss = validate(model,gt_list, ns_list,out_dir, name_list, False)

'''
#gt_list, ns_list, name_list, out_dir = get_test_list('LIVE1',QF, sigma=sigma, is_crop=True)
validate(model,gt_list, ns_list,out_dir, name_list, True)
gt_list, ns_list, name_list, out_dir = get_test_list('B200',QF, sigma=sigma, is_crop=False)
validate(model,gt_list, ns_list,out_dir, name_list, True)
gt_list, ns_list, name_list, out_dir = get_test_list('WIN143',QF, sigma=sigma, is_crop=False)
validate(model,gt_list, ns_list,out_dir, name_list, True)
exit(0)
'''

if multi_gpu:
	p_model = multi_gpu_model(model, gpus=2)
	p_model.compile(optimizer = optimizers.Adam(lr = lr[lr_index]), loss = 'mse')
else:
	model.compile(optimizer = optimizers.Adam(lr = lr[lr_index]), loss = 'mse')

datas_list = []
x_list = []
c_list = []
y_list = []
best_batches = pre_batches
batches = 0
print ('================',datetime.now().strftime('%H:%M:%S'),lr,pre_batches,min_loss,best_batches,'================')
print ('============= version:', lr, batches, best_batches, min_loss, '=============',datetime.now().strftime('%H:%M:%S'))
file_list = os.listdir(images_dir)
print('total images: %d' %(len(file_list)))
for m in range(1,100):
	random.shuffle(file_list)
	img_count = 0
	print('prepareing %d to %d' %(img_count+1, img_count+100))
	for file in file_list:
		s = str.split(file,'.')
		if len(s)!=2 or s[1]!='png':
			continue
		img_count = img_count+1
		img = cv2.imread(images_dir+file)

		#cv2.imwrite(temp_path,img,[1,QF])
		tmp = cv2.imread(LR_dir+s[0]+'.jpg')

		x_step = random.randint(37,57)
		y_step = random.randint(37,57)

		img = img.astype(np.float32)/255.0
		tmp = tmp.astype(np.float32)/255.0
		tmp = np.concatenate([tmp,sigma[0:tmp.shape[0],0:tmp.shape[1],:]],axis=-1)
		for y in range(0,img.shape[0]-data_size,y_step):
			for x in range(0, img.shape[1]-data_size,x_step):
				_x = tmp[y:y+data_size,x:x+data_size]
				if _x.shape[0]!=data_size or _x.shape[1]!=data_size:
					continue				
				_y = img[y:y+data_size,x:x+data_size]
				datas_list.append((_x,_y))
		if img_count % 100 == 0:			
			random.shuffle(datas_list)
			for data in datas_list:
				x_list.append(data[0])
				y_list.append(data[1])
			x_array = np.array(x_list)
			y_array = np.array(y_list)
			datas_list = []
			x_list = []
			y_list = []
			for i in range(0, x_array.shape[0]-batch_size, batch_size):
				batches = batches+1
				if batches <= pre_batches:
					continue
				x_batch = x_array[i:i+batch_size]
				y_batch = y_array[i:i+batch_size]

				if multi_gpu:
					loss = p_model.train_on_batch(x_batch,y_batch)
				else:
					loss = model.train_on_batch(x_batch,y_batch)
				print (batches, loss, end='\r')

				if batches % 10000 == 0:
					#if batches >= 160000:
					#	model.save(model_head+str(batches)+'.h5')	
					if multi_gpu:
						psnr = validate(p_model, gt_list, ns_list,out_dir)
					else:
						psnr = validate(model, gt_list, ns_list,out_dir)
					if evaluation == 'loss':
						if _loss < min_loss:
							best_batches = batches
							model.save(model_head+'best.h5')
							if min_loss-_loss < 1e-6:
								fail_count += 1
							else:
								fail_count = 0
							min_loss = _loss
						else:
							fail_count += 1
					if evaluation == 'psnr':
						_loss = psnr
						if _loss > min_loss:
							best_batches = batches
							model.save(model_head+'best.h5')
							if _loss - min_loss < 1e-3:
								fail_count += 1
							else:
								fail_count = 0
							min_loss = _loss
						else:
							fail_count += 1

					print ('=============', lr, batches, _loss, best_batches, min_loss, '=============',datetime.now().strftime('%H:%M:%S'))
					
					if fail_count >= 4:
					#if batches % 200000 == 0:
						if lr > 2e-6:
							fail_count = 0
							lr = lr / 5
							lr = max(1e-6, lr)
							compile_flag = True
						else:
							exit()

					if compile_flag:
						if multi_gpu:
							p_model.compile(optimizer = optimizers.Adam(lr = lr[lr_index]), loss = 'mse')
						else:
							model.compile(optimizer = optimizers.Adam(lr = lr[lr_index]), loss = 'mse')
						print (batches, lr)
						compile_flag = False
					
			print('prepareing %d to %d' %(img_count+1, img_count+100))	
