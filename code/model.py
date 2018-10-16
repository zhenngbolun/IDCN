from keras import layers, models
from layer import *

def IDCN(h, w, depth, filters, qy, qc, dense_depth=8, growth=32):
	x = layers.Input(shape = (h, w, 6))

	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y

	def dilate_conv(x, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True)(x)
		return y

	def dilate_conv_relu(x, filters, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True, activation='relu')(x)
		return y

	def ref_block(x, filters, dense_depth, growth):
		t = conv_relu(x, growth*2, 3)
		cut = int(dense_depth/2)-2
		for i in range(dense_depth-2):
			rate = 1 + max(0, i-cut)
			y = dilate_conv_relu(t, growth, rate)
			t = layers.Concatenate()([t,y])
		t = conv_relu(t, 64, 1)

		#t1 = layers.Conv2D(64, 1, padding='same', use_bias=True)(t)
		t1 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)
		t1 = rectfied_unit()(t1)
		t1 = implicit_trans(qy)(t1)

		t2 = dilate_conv(t)
		t2 = implicit_trans(qc)(t2)
		t0 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)

		y = layers.Concatenate()([t1,t2])
		y = layers.Conv2D(64,3,padding='same', use_bias=True)(y)
		y = layers.Add()([t0, y])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y

	ec = conv_relu(x, filters, 5)
	m = conv_relu(ec, filters, 3)

	for i in range(depth):
		m = ref_block(m, filters, dense_depth, growth)
	d = conv_relu(m, filters, 3)
	y = conv_relu(d, 3, 5)
	model = models.Model(x,y)
	return model

def ARCNN(h,w):
	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y
	x = layers.Input(shape = (h,w,3))
	t1 = conv_relu(x, 64, 9)
	t2 = conv_relu(t1, 32, 7)
	t3 = conv_relu(t2, 16, 1)
	t4 = conv_relu(t3, 3,  5)
	model = models.Model(x,t4)
	return model

def IDCN_f(h, w, depth, filters, dense_depth=8, growth=32):
	x = layers.Input(shape = (h, w, 6))
	qy = layers.Input(shape = (1,1,64))
	qc = layers.Input(shape = (1,1,64))
	_qy = layers.UpSampling2D((h,w))(qy)
	_qc = layers.UpSampling2D((h,w))(qc)

	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y

	def dilate_conv(x, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True)(x)
		return y

	def dilate_conv_relu(x, filters, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True, activation='relu')(x)
		return y

	def ref_block(x, filters, dense_depth, growth, q1, q2):
		t = conv_relu(x, growth*2, 3)
		cut = int(dense_depth/2)-2
		for i in range(dense_depth-2):
			rate = 1 + max(0, i-cut)
			y = dilate_conv_relu(t, growth, rate)
			t = layers.Concatenate()([t,y])
		t = conv_relu(t, 64, 1)

		#t1 = layers.Conv2D(64, 1, padding='same', use_bias=True)(t)
		t1 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)
		t1 = rectfied_unit()(t1)
		t1 = multi()([t1, q1])
		t1 = IDCT(keep_dims=True)(t1)

		t2 = dilate_conv(t)
		t2 = multi()([t2, q2])
		t2 = IDCT(keep_dims=True)(t2)
		t0 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)

		y = layers.Concatenate()([t1,t2])
		y = layers.Conv2D(64,3,padding='same', use_bias=True)(y)
		y = layers.Add()([t0, y])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y

	ec = conv_relu(x, filters, 5)
	m = conv_relu(ec, filters, 3)

	for i in range(depth):
		m = ref_block(m, filters, dense_depth, growth, _qy, _qc)
	d = conv_relu(m, filters, 3)
	y = conv_relu(d, 3, 5)
	model = models.Model([x,qy,qc],y)
	return model	

#-N and -FC
def network_ab1(h, w, depth, filters, qy, qc, dense_depth=8, growth=32):
	x = layers.Input(shape = (h, w, 3))

	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y

	def dilate_conv(x, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True)(x)
		return y

	def dilate_conv_relu(x, filters, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True, activation='relu')(x)
		return y

	def ref_block(x, filters, dense_depth, growth):
		t = conv_relu(x, growth*2, 3)
		cut = int(dense_depth/2)-2
		for i in range(dense_depth-2):
			rate = 1 + max(0, i-cut)
			y = dilate_conv_relu(t, growth, rate)
			t = layers.Concatenate()([t,y])
		t = conv_relu(t, 64, 1)

		#t1 = layers.Conv2D(64, 1, padding='same', use_bias=True)(t)
		t1 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)
		t1 = rectfied_unit()(t1)
		t1 = implicit_trans(qy)(t1)

		t2 = dilate_conv(t)
		t2 = rectfied_unit()(t2)
		t2 = implicit_trans(qc)(t2)

		t3 = dilate_conv(t)
		t3 = rectfied_unit()(t3)
		t3 = implicit_trans(qc)(t3)

		t0 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)

		y = layers.Concatenate()([t1,t2,t3])
		y = layers.Conv2D(64,3,padding='same', use_bias=True)(y)
		y = layers.Add()([t0, y])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y

	ec = conv_relu(x, filters, 5)
	m = conv_relu(ec, filters, 3)

	for i in range(depth):
		m = ref_block(m, filters, dense_depth, growth)
	d = conv_relu(m, filters, 3)
	y = conv_relu(d, 3, 5)
	model = models.Model(x,y)
	return model

#-N and -SC
def network_ab2(h, w, depth, filters, qy, qc, dense_depth=8, growth=32):
	x = layers.Input(shape = (h, w, 3))

	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y

	def dilate_conv(x, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True)(x)
		return y

	def dilate_conv_relu(x, filters, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True, activation='relu')(x)
		return y

	def ref_block(x, filters, dense_depth, growth):
		t = conv_relu(x, growth*2, 3)
		cut = int(dense_depth/2)-2
		for i in range(dense_depth-2):
			rate = 1 + max(0, i-cut)
			y = dilate_conv_relu(t, growth, rate)
			t = layers.Concatenate()([t,y])
		t = conv_relu(t, 64, 1)

		#t1 = layers.Conv2D(64, 1, padding='same', use_bias=True)(t)
		t1 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)
		t1 = rectfied_unit()(t1)
		t1 = implicit_trans(qy)(t1)

		t2 = dilate_conv(t)
		t2 = implicit_trans(qc)(t2)
		t0 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)

		y = layers.Concatenate()([t1,t2])
		y = layers.Conv2D(64,3,padding='same', use_bias=True)(y)
		y = layers.Add()([t0, y])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y

	ec = conv_relu(x, filters, 5)
	m = conv_relu(ec, filters, 3)

	for i in range(depth):
		m = ref_block(m, filters, dense_depth, growth)
	d = conv_relu(m, filters, 3)
	y = conv_relu(d, 3, 5)
	model = models.Model(x,y)
	return model

#-L and -FC
def network_ab3(h, w, depth, filters, qy, qc, dense_depth=8, growth=32):
	x = layers.Input(shape = (h, w, 6))

	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y

	def dilate_conv(x, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True)(x)
		return y

	def dilate_conv_relu(x, filters, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True, activation='relu')(x)
		return y

	def ref_block(x, filters, dense_depth, growth):
		t = conv_relu(x, growth*2, 3)
		cut = int(dense_depth/2)-2
		for i in range(dense_depth-2):
			rate = 1 + max(0, i-cut)
			y = dilate_conv_relu(t, growth, rate)
			t = layers.Concatenate()([t,y])
		t = conv_relu(t, 64, 1)

		#t1 = layers.Conv2D(64, 1, padding='same', use_bias=True)(t)
		t1 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)
		t1 = rectfied_unit()(t1)
		t1 = implicit_trans(qy)(t1)

		t2 = dilate_conv(t)
		t2 = rectfied_unit()(t2)
		t2 = implicit_trans(qc)(t2)
		t3 = dilate_conv(t)
		t3 = rectfied_unit()(t3)
		t3 = implicit_trans(qc)(t3)

		t0 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)

		y = layers.Concatenate()([t1,t2,t3])
		y = layers.Conv2D(64,3,padding='same', use_bias=True)(y)
		y = layers.Add()([t0, y])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y

	ec = conv_relu(x, filters, 5)
	m = conv_relu(ec, filters, 3)

	for i in range(depth):
		m = ref_block(m, filters, dense_depth, growth)
	d = conv_relu(m, filters, 3)
	y = conv_relu(d, 3, 5)
	model = models.Model(x,y)
	return model

#-L and -SC
def network_ab4(h, w, depth, filters, qy, qc, dense_depth=8, growth=32):
	x = layers.Input(shape = (h, w, 6))

	def conv_relu(x, filters, kernel_size):
		y = layers.Conv2D(filters, kernel_size, padding='same', use_bias=True, activation='relu')(x)
		return y

	def dilate_conv(x, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True)(x)
		return y

	def dilate_conv_relu(x, filters, rate = 2):
		y = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, use_bias=True, activation='relu')(x)
		return y

	def ref_block(x, filters, dense_depth, growth):
		t = conv_relu(x, growth*2, 3)
		cut = int(dense_depth/2)-2
		for i in range(dense_depth-2):
			rate = 1 + max(0, i-cut)
			y = dilate_conv_relu(t, growth, rate)
			t = layers.Concatenate()([t,y])
		t = conv_relu(t, 64, 1)

		#t1 = layers.Conv2D(64, 1, padding='same', use_bias=True)(t)
		t1 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)
		t1 = rectfied_unit()(t1)
		t1 = implicit_trans(qy)(t1)

		t2 = dilate_conv(t)
		t2 = implicit_trans(qc)(t2)
		t0 = layers.Conv2D(64, 3, padding='same', use_bias=True)(t)

		y = layers.Concatenate()([t1,t2])
		y = layers.Conv2D(64,3,padding='same', use_bias=True)(y)
		y = layers.Add()([t0, y])
		y = layers.Lambda(lambda x:x*0.1)(y)
		y = layers.Add()([x,y])
		return y

	ec = conv_relu(x, filters, 5)
	m = conv_relu(ec, filters, 3)

	for i in range(depth):
		m = ref_block(m, filters, dense_depth, growth)
	d = conv_relu(m, filters, 3)
	y = conv_relu(d, 3, 5)
	model = models.Model(x,y)
	return model