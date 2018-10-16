import tensorflow as tf
import os, cv2, random, keras
import numpy as np
from model import *
from keras import optimizers
from math import log10
from datas import *

ql=5
qh=20
length = qh-ql+1
_sigma = np.zeros([length,16,16,3])
sigma_bl, sigma_gl, sigma_rl = generate_sigma(ql)
sigma_bh, sigma_gh, sigma_rh = generate_sigma(qh)
sigma = np.zeros([1400,1400,3])

sl = int(5000/ql)
sh = int(5000/qh)
for i in range(length):
	_q = i+ql
	_s = float(int(5000/_q))
	_sigma[i,:,:,0] = sigma_bl*(_s-sh)/(sl-sh)+sigma_bh*(sl-_s)/(sl-sh)
	_sigma[i,:,:,1] = sigma_gl*(_s-sh)/(sl-sh)+sigma_gh*(sl-_s)/(sl-sh)
	_sigma[i,:,:,2] = sigma_rl*(_s-sh)/(sl-sh)+sigma_rh*(sl-_s)/(sl-sh)

depth = 8
filters = 64
def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config = config)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
keras.backend.tensorflow_backend.set_session(get_session())

images_dir = 'datas/LIVE1/'

weights = 'network20_ab4_best.h5'

for QF in range(20,21):
	print 'QF = '+str(QF)+':'
	qy = get_table(luminance_quant_table, QF)
	#qy = qy.reshape(1,1,1,64)
	qc = get_table(chrominance_quant_table, QF)
	#qc = qc.reshape(1,1,1,64)
	for y in range(sigma.shape[0]):
		for x in range(sigma.shape[1]):
			_y = y%16
			_x = x%16
			sigma[y,x] = _sigma[QF-ql,_y,_x]

	img_count = 0
	avg_psnr = 0
	file_list = os.listdir(images_dir)
	for file in file_list:
		s = str.split(file,'.')
		if len(s)!=2 or s[1]!='bmp':
			continue
		img_count = img_count+1
		img = cv2.imread(images_dir+file)
		w = img.shape[1]
		h = img.shape[0]
		img = img[0:h,0:w]
		cv2.imwrite('temp.jpg',img,[1,QF])
		tmp = cv2.imread('temp.jpg')

		img = img.astype(np.float32)/255.0
		tmp = tmp.astype(np.float32)/255.0

		tmp = np.concatenate([tmp,sigma[0:tmp.shape[0],0:tmp.shape[1],:]],axis=-1)
		tmp = tmp.reshape(1,tmp.shape[0],tmp.shape[1],tmp.shape[2])
		#input_shape = (h,w,1)
		if version == 'IDCN':
			model = IDCN(h,w,depth,filters,qy,qc,8,64)
			weights = 'IDCN-10.h5'#'IDCN-20.h5'
		elif version == 'IDCN-f':
			model = IDCN_f(h,w,depth,filters,qy,qc,8,64)
			weights = 'IDCN-f.h5'
		model.load_weights(weights)
		y = model.predict(tmp)
		_y = y[0,:,:]*255.0
		_y[_y>255]=255
		_y = _y + 0.5
		_y = _y.astype(np.uint8)
		cv2.imwrite('images/'+file, _y)
		_y = _y.astype(np.float32)/255.0
		mse = np.mean(np.square(_y-img))
		psnr = 10*log10(1/mse)
		print img_count, file, psnr
		avg_psnr = avg_psnr+psnr
	print avg_psnr/img_count
