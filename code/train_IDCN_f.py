import tensorflow as tf
import os, cv2, random, keras, random
import numpy as np
from model import *
from keras import optimizers
from keras.utils import multi_gpu_model
from math import log10
from utils import *
from datetime import datetime

multi_gpu = False
if multi_gpu:	
	print ('use multi gpu mode!')
	os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
else:
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
keras.backend.tensorflow_backend.set_session(get_session())

images_dir = 'F:/dataset/DIV2K/DIV2K_train_HR/'

datas_list = []
x_list = []
y_list = []
qy_list = []
qc_list = []
img_count = 0
batches = 0
min_psnr = 0
best_batches = 0
fail_count = 0
data_size = 43
batch_size = 16
lr = 1e-4
compile_flag = False

ql=5
qh=20
length = qh-ql+1
_sigma = np.zeros([length,16,16,1])
sigma_bl, sigma_gl, sigma_rl = generate_sigma(ql)
sigma_bh, sigma_gh, sigma_rh = generate_sigma(qh)
sigma_l = np.sqrt(sigma_bl**2 + sigma_gl**2 + sigma_rl**2)
sigma_h = np.sqrt(sigma_bh**2 + sigma_gh**2 + sigma_rh**2)
sl = int(5000/ql)
sh = int(5000/qh)
for i in range(length):
	_q = i+ql
	_s = float(int(5000/_q))
	_sigma[i,:,:,0] = sigma_l*(_s-sh)/(sl-sh)+sigma_h*(sl-_s)/(sl-sh)

PyramidCells = ((3,2,1,1,1,1),)
model = IDCN_f(64, 8, PyramidCells, 64)
model.load_weights('IDCN_f_5_20_best.h5')
sigma = np.zeros([length, 2400,2400,1])
for l in range(length):
	for y in range(sigma.shape[0]):
		for x in range(sigma.shape[1]):
			_y = y%16
			_x = x%16
			sigma[l,y,x,0] = _sigma[l,_y,_x,0]

model_head = 'weights/IDCN_f_'+str(ql)+'_'+str(qh)+'_'
temp_path = 'temp_.jpg'

if multi_gpu:
	p_model = multi_gpu_model(model, gpus=2)
	p_model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mse')
else:
	model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mse')


file_list = os.listdir('F:\\dataset\\BSDS500\\images\\test\\')
for file in file_list:
	s = str.split(file,'.')
	if len(s)!=2 or s[1]!='jpg':
		continue
	img_count = img_count+1
	img = cv2.imread('F:\\dataset\\BSDS500\\images\\test\\'+file)
	QF = random.randint(ql,qh)
	index = QF-ql
	_qy = get_table(luminance_quant_table, QF)
	_qc = get_table(chrominance_quant_table, QF)
	_qy = _qy.reshape(1,1,64)
	_qc = _qc.reshape(1,1,64)
	cv2.imwrite(temp_path,img,[1,QF])
	tmp = cv2.imread(temp_path)
	
	img = img.astype(np.float32)/255.0
	tmp = tmp.astype(np.float32)/255.0
	#network>=v7
	tmp = np.concatenate([tmp,sigma[index,0:tmp.shape[0],0:tmp.shape[1],:]],axis=-1)

	for y in range(0,img.shape[0]-data_size,data_size):
		for x in range(0, img.shape[1]-data_size,data_size):
			_x = tmp[y:y+data_size,x:x+data_size]
			if _x.shape[0]!=data_size or _x.shape[1]!=data_size:
				continue
			_y = img[y:y+data_size,x:x+data_size]
			x_list.append(_x)
			y_list.append(_y)
			qy_list.append(_qy)
			qc_list.append(_qc)
testx_array = np.array(x_list)
testy_array = np.array(y_list)
testqy_array = np.array(qy_list)
testqc_array = np.array(qc_list)
print (testx_array.shape)
datas_list = []
x_list = []
y_list = []
qy_list = []
qc_list = []

print ('================',datetime.now().strftime('%H:%M:%S'),lr,batches,min_psnr,best_batches,'================')
print ('=============', lr, batches, best_batches, min_psnr, '=============',datetime.now().strftime('%H:%M:%S'))
file_list = os.listdir(images_dir)
for m in range(100):
	random.shuffle(file_list)
	for file in file_list:
		s = str.split(file,'.')
		if len(s)!=2 or s[1]!='png':
			continue
		img_count = img_count+1
		img = cv2.imread(images_dir+file)
		QF = random.randint(ql,qh)
		cv2.imwrite(temp_path,img,[1,QF])
		tmp = cv2.imread(temp_path)
		qy = get_table(luminance_quant_table, QF)
		qc = get_table(chrominance_quant_table, QF)
		qy = qy.reshape(1,1,64)
		qc = qc.reshape(1,1,64)
		x_step = random.randint(37,57)
		y_step = random.randint(37,57)
		img = img.astype(np.float32)/255.0
		tmp = tmp.astype(np.float32)/255.0

		tmp = np.concatenate([tmp,sigma[QF-ql,0:tmp.shape[0],0:tmp.shape[1],:]],axis=-1)
		for y in range(0,img.shape[0]-data_size,y_step):
			for x in range(0, img.shape[1]-data_size,x_step):
				_x = tmp[y:y+data_size,x:x+data_size]
				if _x.shape[0]!=data_size or _x.shape[1]!=data_size:
					continue
				_y = img[y:y+data_size,x:x+data_size]
				datas_list.append((_x,_y,qy,qc))
		if img_count % 50 == 0:
			random.shuffle(datas_list)
			for data in datas_list:
				x_list.append(data[0])
				y_list.append(data[1])
				qy_list.append(data[2])
				qc_list.append(data[3])
			x_array = np.array(x_list)
			y_array = np.array(y_list)
			qy_array = np.array(qy_list)
			qc_array = np.array(qc_list)
			datas_list = []
			x_list = []
			y_list = []
			qy_list = []
			qc_list = []
			for i in range(0, x_array.shape[0]-batch_size, batch_size):
				batches = batches+1
				x_batch = x_array[i:i+batch_size]
				qy_batch = qy_array[i:i+batch_size]
				qc_batch = qc_array[i:i+batch_size]
				y_batch = y_array[i:i+batch_size]
				if multi_gpu:
					loss = p_model.train_on_batch([x_batch,qy_batch,qc_batch],y_batch)
				else:
					loss = model.train_on_batch([x_batch,qy_batch,qc_batch],y_batch)
				print (batches, loss, end='\r')

				if batches % 5000 == 0:
					if batches >= 160000:
						model.save(model_head+str(batches)+'.h5')
						
					if multi_gpu:
						y = p_model.predict([testx_array,testqy_array,testqc_array])
					else:
						y = model.predict([testx_array,testqy_array,testqc_array])
					_psnr = 10*log10(1/np.mean(np.square((y) - (testy_array))))
					
					if _psnr > min_psnr:
						best_batches = batches
						model.save(model_head+'best.h5')
						if _psnr-min_psnr < 1e-3:
							fail_count += 1
						else:
							fail_count = 0
						min_psnr = _psnr
					else:
						fail_count += 1

					print ('=============', lr, batches, _psnr, best_batches, min_psnr, '=============',datetime.now().strftime('%H:%M:%S'))
					if fail_count >= 3:
						if lr > 2e-6:
							fail_count = 0
							lr = lr / 5
							lr = max(1e-6, lr)
							compile_flag = True
						elif fail_count == 5:
							exit()
					if compile_flag:
						if multi_gpu:
							p_model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mse')
						else:
							model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mse')
						print (batches, lr)
						compile_flag = False
						
				#if batches % 5000 == 0 and lr > 1e-6:
				#	lr = lr / 2
				#	model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mse')