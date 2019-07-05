from keras import layers
from keras.models import Model
from keras import backend as K
from core_layers import *

def conv_relu(x, filters, kernel, use_bias = True, dilation_rate=1):
	if dilation_rate == 0:
		y = layers.Conv2D(filters,1,padding='same',use_bias=use_bias,
			activation='relu')(x)
	else:
		y = layers.Conv2D(filters,kernel,padding='same',use_bias=use_bias,
			dilation_rate=dilation_rate,
			activation='relu')(x)
	return y

def conv(x, filters, kernel, use_bias=True, dilation_rate=1):
	y = layers.Conv2D(filters,kernel,padding='same',use_bias=use_bias,
		dilation_rate=dilation_rate)(x)
	return y

def pyramid(x, filters, Pyramid_Cells):
	def pyramid_cell(x, filters, dilation_rates):
		for i in range(len(dilation_rates)):
			dilation_rate = dilation_rates[i]
			if i==0:
				t = conv_relu(x,filters,3,dilation_rate=dilation_rate)
				_t = layers.Concatenate(axis=-1)([x,t])
			else:
				t = conv_relu(_t,filters,3,dilation_rate=dilation_rate)
				_t = layers.Concatenate(axis=-1)([_t,t])
		return _t
	concat_list = []
	t = conv_relu(x,filters*2,3)
	for i in range(len(Pyramid_Cells)):
		if i == 0:
			_t = pyramid_cell(t,filters,Pyramid_Cells[i])
		else:
			_t = pyramid_cell(_t,filters,Pyramid_Cells[i])
		_t = conv_relu(_t,filters,1)
		concat_list.append(_t)		
	if len(concat_list) == 1:
		return _t
	else:
		y = layers.Concatenate(axis=-1)(concat_list)
		return y

def IDCN(nFilters, nPyramids, Pyramid_Cells, nPyramidFilters, qy, qc, Restriction='hard'):
	output_list = []
	def dual_domain_block(x,nFilters,Pyramid_Cells,nPyramidFilters,qy,qc,Restriction):
		_t = pyramid(x, nPyramidFilters, Pyramid_Cells)
		
		_ty = conv(_t, nFilters, 3)
		_tc = conv(_t, nFilters, 3, dilation_rate=2)
		if Restriction == 'RaRe':
			_ty = RaRe(r=0.75)(_ty)
			_tc = RaRe(r=0.75)(_tc)
		if Restriction == 'soft':
			_ty = SoftThreshing(th=0.5)(_ty)
			_tc = SoftThreshing(th=0.5)(_tc)
		if Restriction == 'hard':
			_ty = HardThreshing(th=0.5)(_ty)
			#_tc = HardThreshing(th=0.75)(_tc)
		output_list.append(_ty)
		_ty = implicit_trans(q=qy)(_ty)
		_tc = implicit_trans(q=qc)(_tc)
		_tp = conv(_t, nFilters, 3)
		_td = layers.Concatenate(axis=-1)([_ty,_tc])
		_td = conv(_td, nFilters, 3)
		y = layers.Add()([_td,_tp])
		
		#y = conv(_t, nFilters, 3)
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y
	
	x = layers.Input(shape=(None, None, 4))
	t = conv_relu(x,nFilters,5)
	t = conv_relu(t,nFilters,3)
	for i in range(nPyramids):
		t = dual_domain_block(t,nFilters,Pyramid_Cells,nPyramidFilters,qy,qc,Restriction)
	t = conv_relu(t,nFilters,3)
	y = conv_relu(t,3,5)
	output_list.append(y)
	return models.Model(x,output_list)

def IDCN_Y(nFilters, nPyramids, Pyramid_Cells, nPyramidFilters, qy, Restriction='hard'):
	def dual_domain_block(x,nFilters,Pyramid_Cells,nPyramidFilters,qy,Restriction):
		_t = pyramid(x, nPyramidFilters, Pyramid_Cells)
		
		_ty = conv(_t, nFilters, 3)
		if Restriction == 'RaRe':
			_ty = RaRe(r=0.75)(_ty)
		if Restriction == 'soft':
			_ty = SoftThreshing(th=0.5)(_ty)
		if Restriction == 'hard':
			_ty = HardThreshing(th=0.5)(_ty)
		_ty = implicit_trans(q=qy)(_ty)
		_tp = conv(_t, nFilters, 3)
		_td = conv(_ty, nFilters, 3)
		y = layers.Add()([_td,_tp])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y
	
	x = layers.Input(shape=(None, None, 2))
	t = conv_relu(x,nFilters,5)
	t = conv_relu(t,nFilters,3)
	for i in range(nPyramids):
		t = dual_domain_block(t,nFilters,Pyramid_Cells,nPyramidFilters,qy,Restriction)
	t = conv_relu(t,nFilters,3)
	y = conv_relu(t,1,5)
	return models.Model(x,y)

def IDCN_f(nFilters, nPyramids, Pyramid_Cells, nPyramidFilters, Restriction='hard'):
	def dual_domain_block(x,nFilters,Pyramid_Cells,nPyramidFilters,qy,qc,Restriction):
		_t = pyramid(x, nPyramidFilters, Pyramid_Cells)
		
		_ty = conv(_t, nFilters, 3)
		_tc = conv(_t, nFilters, 3, dilation_rate=2)
		if Restriction == 'RaRe':
			_ty = RaRe(r=0.75)(_ty)
			_tc = RaRe(r=0.75)(_tc)
		if Restriction == 'soft':
			_ty = SoftThreshing(th=0.5)(_ty)
			_tc = SoftThreshing(th=0.5)(_tc)
		if Restriction == 'hard':
			_ty = HardThreshing(th=0.5)(_ty)
			#_tc = HardThreshing(th=0.75)(_tc)
		_ty = layers.Multiply()([_ty,qy])
		_tc = layers.Multiply()([_tc,qc])
		_ty = IDCT(True)(_ty)
		_tc = IDCT(True)(_tc)
		_tp = conv(_t, nFilters, 3)
		_td = layers.Concatenate(axis=-1)([_ty,_tc])
		_td = conv(_td, nFilters, 3)
		
		y = layers.Add()([_td,_tp])
		
		#y = conv(_t, nFilters, 3)
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y
	
	x = layers.Input(shape=(None, None, 4))
	qy = layers.Input(shape=(1,1,64))
	qc = layers.Input(shape=(1,1,64))
	t = conv_relu(x,nFilters,5)
	t = conv_relu(t,nFilters,3)
	for i in range(nPyramids):
		t = dual_domain_block(t,nFilters,Pyramid_Cells,nPyramidFilters,qy,qc,Restriction)
	t = conv_relu(t,nFilters,3)
	y = conv_relu(t,3,5)
	return models.Model([x,qy,qc],y)

def SNet(channel=1):
	x = layers.Input(shape=(None, None, channel))
	t = conv_relu(x, 256, 5)
	t = conv_relu(t, 256, 3)
	for i in range(8):
		_t = conv_relu(t, 256, 3)
		_t = conv(_t, 256, 3)
		_t = layers.Lambda(lambda x:x*0.1)(_t)
		t = layers.Add()([t, _t])
	t = conv_relu(t, 256, 3)
	y = conv_relu(t, channel, 5)
	model = Model(x, y)
	return model

def RDN_JPEG(N=8, channel = 1, cp=True):
	concat_list = []
	def rdb(x):
		if cp:
			t = DenseLayer(nFilters=64,depth=2,useBias=False)(x)
		else:
			for i in range(8-1):
				if i == 0:
					t = conv_relu(x, 64, 3, use_bias=False)
					t = layers.Concatenate(axis=-1)([x,t])
				else:
					_t = conv_relu(t, 64, 3, use_bias=False)
					t = layers.Concatenate(axis=-1)([_t,t])

		t = conv(t, 64, 1)
		t = layers.Add()([x, t])
		
		return t

	x = layers.Input(shape=(None, None, channel))
	t1 = conv_relu(x, 64, 3)
	t2 = conv_relu(t1, 64, 3)
	for i in range(N):
		if i == 0:
			t = rdb(t2)
		else:
			t = rdb(t)
		concat_list.append(t)
	t = layers.Concatenate(axis=-1)(concat_list)
	t = conv(t, 64, 1, use_bias=False)
	t = conv(t, 64, 3, use_bias=False)
	t = layers.Add()([t1, t])
	t = conv(t, channel, 3, use_bias=False)	
	y = layers.Add()([x, t])
	model = Model(x, y)
	return model

def MemNet(channel = 1, training=True):
	def bn_relu_conv(x, nFilters, kernel):
		t = layers.BatchNormalization(axis=-1)(x)
		t = layers.Activation('relu')(t)
		t = layers.Conv2D(nFilters,kernel,padding='same',use_bias=False)(t)
		return t

	def mem_block(x,_list):
		concat_list = []
		for i in range(6):
			if i == 0:
				_t = bn_relu_conv(x, 64, 3)
				_t = bn_relu_conv(_t, 64, 3)
				t = layers.Add()([x,_t])
			else:
				_t = bn_relu_conv(t, 64, 3)
				_t = bn_relu_conv(_t, 64, 3)
				t = layers.Add()([t,_t])
			concat_list.append(t)
		concat_list += _list
		t = layers.Concatenate(axis=-1)(concat_list)
		y = bn_relu_conv(t, 64, 1)
		return y

	def dec(x):
		y = bn_relu_conv(x, channel, 3)
		return y

	add_list = []
	out_list = []
	_list = []
	x = layers.Input(shape=(None,None,channel))
	f = bn_relu_conv(x, 64, 3)
	_list.append(f)
	for i in range(6):
		if i == 0:
			m = mem_block(f,_list)
			_list.append(m)
			rec = dec(m)
			rec = layers.Add()([x,rec])
			out_list.append(rec)
			rec = ScaleLayer(s=(1.0/6))(rec)
			add_list.append(rec)
		else:
			m = mem_block(m,_list)
			_list.append(m)
			rec = dec(m)
			rec = layers.Add()([x,rec])
			out_list.append(rec)
			rec = ScaleLayer(s=(1.0/6))(rec)
			add_list.append(rec)
	y = layers.Add()(add_list)
	out_list.append(y)
	if training:
		model = models.Model(x,out_list)
		loss_weights = []
		for i in range(7):
			loss_weights.append(1.0/7)
			#model.compile(optimizer=, loss='mse', loss_weights=loss_weights)
		return model
	else:
		return models.Model(x,y)

def MWCNN(channel=1):
	x = layers.Input(shape=(None,None,channel))
	d1 = DWT()(x)
	d1 = conv_relu(d1, 160, 3)
	for i in range(3):
		d1 = conv_relu(d1, 160, 3)
	d2 = DWT()(d1)
	for i in range(4):
		d2 = conv_relu(d2, 256, 3)
	d3 = DWT()(d2)
	for i in range(7):
		d3 = conv_relu(d3, 256, 3)
	d3 = conv_relu(d3, 1024, 3)
	d3 = IWT()(d3)
	d2 = layers.Add()([d2,d3])
	for i in range(3):
		d2 = conv_relu(d2, 256, 3)
	d2 = conv_relu(d2, 640, 3)
	d2 = IWT()(d2)
	d1 = layers.Add()([d1,d2])
	for i in range(3):
		d1 = conv_relu(d1, 160, 3)
	d1 = conv(d1, 4*channel,3)
	d1 = IWT()(d1)
	y = layers.Add()([x,d1])
	model = models.Model(x,y)
	return model
	
def ARCNN(channel=1):
	x = layers.Input(shape=(None, None, channel))
	t = conv_relu(x, 64, 9)
	t = conv_relu(t, 32, 7)
	t = conv_relu(t, 16, 1)
	y = conv(t,channel,5)
	return models.Model(x,y)

def D_SDNet(channel=1):
	def conv_bn_relu(x, nFilters, kernel):
		t = layers.Conv2D(nFilters,kernel,padding='same',use_bias=False)(x)
		t = layers.BatchNormalization(axis=-1)(t)
		t = layers.Activation('relu')(t)		
		return t

	x = layers.Input(shape=(None,None,channel))
	t = Space2Depth(2)(x)
	for i in range(19):
		t = conv_bn_relu(t,64,3)
	t = conv(t,channel*4,3)
	t = Depth2Space(2)(t)
	y = layers.Add()([x,t])
	return models.Model(x,y)

def P_SDNet(channel=1):
	def conv_bn_relu(x, nFilters, kernel):
		t = layers.Conv2D(nFilters,kernel,padding='same',use_bias=False)(x)
		t = layers.BatchNormalization(axis=-1)(t)
		t = layers.Activation('relu')(t)		
		return t

	x = layers.Input(shape=(None,None,3))
	t = DWT()(x)
	for i in range(19):
		t = conv_bn_relu(t,64,3)
	t = conv(t,channel*4,3)
	t = IWT()(t)
	y = layers.Add()([x,t])
	return models.Model(x,y)
