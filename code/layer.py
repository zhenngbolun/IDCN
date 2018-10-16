from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from math import pi, cos, sqrt
import keras

class DCT(Layer):
	def __init__(self, **kwargs):
		super(DCT, self).__init__(**kwargs)
	
	def build(self, input_shape):
		dct_kernel = np.zeros([8,8,1,64])
		for i in range(8):
			for j in range(8):
				index = j*8+i
				for x in range(8):
					for y in range(8):
						t = 0.25*cos((2*x+1)*pi*i/16)*cos((2*y+1)*pi*j/16)
						t = t/sqrt(2) if i==0 else t
						t = t/sqrt(2) if j==0 else t
						dct_kernel[y,x,0,index]=t
		self.dct_kernel = K.variable(value = dct_kernel)

	def call(self, inputs):
		y = K.conv2d(inputs, self.dct_kernel)
		return y

	def compute_output_shape(self, input_shape):
		return(None, input_shape[1]-7,input_shape[2]-7,64)

class IDCT(Layer):
	"""docstring for IDCT"""
	def __init__(self, 
				 keep_dims = False,
				 **kwargs):
		super(IDCT, self).__init__(**kwargs)
		self.keep_dims = keep_dims

	def build(self, input_shape):
		if self.keep_dims:
			dct_kernel = np.zeros([1,1,64,64])
			for x in range(8):
				for y in range(8):
					index = y*8+x
					for i in range(8):
						for j in range(8):
							t = 0.25*cos((2*x+1)*pi*i/16)*cos((2*y+1)*pi*j/16)
							t = t/sqrt(2) if i==0 else t
							t = t/sqrt(2) if j==0 else t
							dct_kernel[0,0,j*8+i,index]=t
		else:
			dct_kernel = np.zeros([8,8,1,64])
			for i in range(8):
				for j in range(8):
					index = j*8+i
					for x in range(8):
						for y in range(8):
							t = 0.25*cos((2*x+1)*pi*i/16)*cos((2*y+1)*pi*j/16)
							t = t/sqrt(2) if i==0 else t
							t = t/sqrt(2) if j==0 else t
							dct_kernel[y,x,0,index]=t
		self.dct_kernel = K.variable(value = dct_kernel)

		scale_kernel = np.zeros([input_shape[1]+7,input_shape[2]+7,1])
		for i in range(input_shape[2]+7):
			for j in range(input_shape[1]+7):
				_i = np.min([i+1,8,input_shape[2]+7-i])
				_j = np.min([j+1,8,input_shape[1]+7-j])
				scale_kernel[j,i] = 1.0/(_i*_j)
		self.scale_kernel = K.variable(value = scale_kernel)
	def call(self, inputs):
		shape = K.shape(inputs)
		if self.keep_dims:
			y = K.conv2d(inputs, self.dct_kernel)
		else:
			y = K.conv2d_transpose(inputs,self.dct_kernel,output_shape=(shape[0],shape[1]+7,shape[2]+7,1))
			y = y*self.scale_kernel
		return y

	def compute_output_shape(self, input_shape):
		if self.keep_dims:
			return input_shape
		else:
			return(input_shape[0], input_shape[1]+7, input_shape[2]+7, int(input_shape[3]/64))

class PixelShuffleBack(Layer):
	def __init__(self, ratio, **kwargs):
		super(PixelShuffleBack, self).__init__(**kwargs)
		self.ratio = ratio

	def call(self, inputs):
		return tf.depth_to_space(inputs, self.ratio)

	def compute_output_shape(self, input_shape):
		return(None, input_shape[1]*self.ratio, input_shape[2]*self.ratio, int(input_shape[3]/self.ratio**2))

class Scale(Layer):
	def __init__(self, **kwargs):
		super(Scale, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = K.variable(keras.initializers.Constant(value=0.5)((1,)))
		self.trainable_weights=self.alpha
	def call(self, inputs):
		y = inputs*self.alpha
		return y

	def compute_output_shape(self, input_shape):
		return input_shape

class multi(Layer):
	def __init__(self, **kwargs):
		super(multi, self).__init__(**kwargs)
	def call(self, inputs):
		a = inputs[0]
		b = inputs[1]
		return a*b
	def compute_output_shape(self, input_shape):
		return input_shape

class implicit_trans(Layer):
	def __init__(self, q, **kwargs):
		self.q = q
		super(implicit_trans, self).__init__(**kwargs)

	def build(self, input_shape):
		conv_shape = (1,1,64,64)
		kernel = np.zeros(conv_shape)
		r1 = sqrt(1/8)
		r2 = sqrt(2/8)
		for i in range(8):
			_u = 2*i+1
			for j in range(8):
				_v = 2*j+1
				index = i*8+j
				for u in range(8):
					for v in range(8):
						index2 = u*8+v
						t = self.q[u,v]*cos(_u*u*pi/16)*cos(_v*v*pi/16)
						t = t*r1 if u==0 else t*r2
						t = t*r1 if v==0 else t*r2
						kernel[0,0,index2,index] = t
		self.kernel = K.variable(value = kernel, dtype = 'float32')

	def call(self, inputs):
		y = K.conv2d(inputs,
					 self.kernel,
					 padding = 'same',
					 data_format='channels_last')
		return y

	def compute_output_shape(self, input_shape):
		return input_shape

class rectfied_unit(Layer):
	def __init__(self, **kwargs):
		super(rectfied_unit, self).__init__(**kwargs)

	def call(self, inputs):
		y = K.maximum(inputs,-0.5)
		y = K.minimum(y, 0.5)
		return y

	def compute_output_shape(self, input_shape):
		return input_shape 