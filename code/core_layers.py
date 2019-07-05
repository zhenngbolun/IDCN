import tensorflow as tf
from keras import backend as k
from keras.utils import conv_utils
import numpy as np
from keras import layers, models, activations, initializers
from math import cos, pi, sqrt
from keras.regularizers import l2

class Space2Depth(layers.Layer):
	def __init__(self, scale, **kwargs):
		super(Space2Depth, self).__init__(**kwargs)
		self.scale = scale

	def call(self, inputs, **kwargs):
		return tf.space_to_depth(inputs, self.scale)

	def compute_output_shape(self, input_shape):
		return (None, None, None, input_shape[-1]*(self.scale**2))

class Depth2Space(layers.Layer):
	def __init__(self, scale, **kwargs):
		super(Depth2Space, self).__init__(**kwargs)
		self.scale = scale
	def call(self, inputs, **kwargs):
		return tf.depth_to_space(inputs, self.scale)

	def compute_output_shape(self, input_shape):
		return (None, None, None, int(input_shape[-1]/(self.scale**2)))

class UpSample(layers.Layer):
	def __init__(self, scale, **kwargs):
		super(UpSample, self).__init__(**kwargs)
		self.scale = scale

	def build(self, input_shape):
		input_dim = input_shape[-1]
		kernel_shape = (3,3) + (input_dim, input_dim*self.scale*self.scale)	
		print(kernel_shape)
		self.kernel = self.add_weight(
			shape = kernel_shape,
			initializer = initializers.get('glorot_uniform'),
			name = 'upsample_conv')
	
	def call(self, inputs):
		t = k.conv2d(
			inputs,
			self.kernel,
			padding='same'
		)
		t = tf.depth_to_space(t, self.scale)
		return t
	def compute_output_shape(self, input_shape):
		if input_shape[1] != None and input_shape[2] != None:
			return (None, input_shape[1]*self.scale, input_shape[2]*self.scale, input_shape[3])
		else:
			return input_shape

class Sample(layers.Layer):
	def __init__(self, window, **kwargs):
		super(Sample, self).__init__(**kwargs)
		self.window = window

	def build(self, input_shape):
		window = self.window
		kernel = np.zeros((window, window, 1, window*window),dtype=np.float32)
		for i in range(window*window):
			kernel[int(i/window),int(i%window),0,i]=1
		self.kernel = k.variable(value=kernel)
	def call(self, inputs, **kwargs):
		y = k.conv2d(inputs, self.kernel)
		return y

	def compute_output_shape(self, input_shape):
		return (None, None, None, self.window*self.window)
		
class Dct1D(layers.Layer):
	def __init__(self, **kwargs):
		super(Dct1D, self).__init__(**kwargs)
	def build(self, input_shape):
		N = input_shape[-1]
		kernel = np.zeros((1,1,N,N),dtype=np.float32)
		for u in range(N):
			for i in range(N):
				kernel[0,0,i,u] = cos((2*i+1)*u*pi/2/N)
			if u == 0:
				kernel[0,0,:,u] *= sqrt(1/N)
			else:
				kernel[0,0,:,u] *= sqrt(2/N)
		self.kernel = k.variable(value=kernel)
	def call(self, inputs, **kwargs):
		y = k.conv2d(inputs, self.kernel)
		return y

	def compute_output_shape(self, input_shape):
		return input_shape

class iDct1D(layers.Layer):
	def __init__(self, **kwargs):
		super(iDct1D, self).__init__(**kwargs)
	def build(self, input_shape):
		N = input_shape[-1]
		kernel = np.zeros((1,1,N,N),dtype=np.float32)
		for u in range(N):
			for i in range(N):
				kernel[0,0,u,i] = cos((2*i+1)*u*pi/2/N)
			if u == 0:
				kernel[0,0,u,:] *= sqrt(1/N)
			else:
				kernel[0,0,u,:] *= sqrt(2/N)
		self.kernel = k.variable(value=kernel)
	def call(self, inputs, **kwargs):
		y = k.conv2d(inputs, self.kernel)
		return y

	def compute_output_shape(self, input_shape):
		return input_shape

class IDCT(layers.Layer):
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
			scale_kernel = np.zeros([input_shape[1]+7,input_shape[2]+7,1])
			for i in range(input_shape[2]+7):
				for j in range(input_shape[1]+7):
					_i = np.min([i+1,8,input_shape[2]+7-i])
					_j = np.min([j+1,8,input_shape[1]+7-j])
					scale_kernel[j,i] = 1.0/(_i*_j)
			self.scale_kernel = k.variable(value = scale_kernel)
		self.dct_kernel = k.variable(value = dct_kernel)

		
	def call(self, inputs):
		shape = k.shape(inputs)
		if self.keep_dims:
			y = k.conv2d(inputs, self.dct_kernel)
		else:
			y = k.conv2d_transpose(inputs,self.dct_kernel,output_shape=(shape[0],shape[1]+7,shape[2]+7,1))
			y = y*self.scale_kernel
		return y

	def compute_output_shape(self, input_shape):
		if self.keep_dims:
			return input_shape
		else:
			return(input_shape[0], input_shape[1]+7, input_shape[2]+7, int(input_shape[3]/64))

class HardThreshing(layers.Layer):
	def __init__(self, th = 0.5, **kwargs):
		super(HardThreshing, self).__init__(**kwargs)
		if th > 0:
			self.th_max = th
			self.th_min = th*(-1)
		else:
			self.th_max = th*(-1)
			self.th_min = th

	def call(self, inputs, **kwargs):
		y = k.maximum(inputs,self.th_min)
		y = k.minimum(y, self.th_max)
		return y

	def compute_output_shape(self, input_shape):
		return input_shape

class SoftThreshing(layers.Layer):
	def __init__(self, alpha = 0.1, th = 0.5, **kwargs):
		super(SoftThreshing, self).__init__(**kwargs)
		self.alpha = alpha
		if th > 0:
			self.th_max = th
			self.th_min = th*(-1)
		else:
			self.th_max = th*(-1)
			self.th_min = th

	def call(self, inputs, **kwargs):
		y_max = k.maximum(inputs,self.th_min)
		t = (inputs - y_max) * self.alpha + y_max
		y_min = k.minimum(t, self.th_max)
		y = (t - y_min) * self.alpha + y_min
		return y

	def compute_output_shape(self, input_shape):
		return input_shape

class DWT(layers.Layer):
	def __init__(self, **kwargs):
		super(DWT, self).__init__(**kwargs)

	def call(self, inputs, **kwargs):
		x01 = inputs[:,0::2,:,:] / 4.0
		x02 = inputs[:,1::2,:,:] / 4.0
		x1 = x01[:,:,0::2,:]
		x2 = x01[:,:,1::2,:]
		x3 = x02[:,:,0::2,:]
		x4 = x02[:,:,1::2,:]
		y1 = x1+x2+x3+x4
		y2 = x1-x2+x3-x4
		y3 = x1+x2-x3-x4
		y4 = x1-x2-x3+x4
		y = k.concatenate([y1,y2,y3,y4],axis=-1)
		return y

	def compute_output_shape(self, input_shape):
		c = input_shape[-1]*4
		if(input_shape[1] != None and input_shape[2] != None):
			return (input_shape[0], input_shape[1] >> 1, input_shape[2] >> 1, c)
		else:
			return (None, None, None, c)

class IWT(layers.Layer):
	def __init__(self, **kwargs):
		super(IWT, self).__init__(**kwargs)

	def build(self, input_shape):
		c = input_shape[-1]
		out_c = c >> 2
		kernel = np.zeros((1,1,c,c),dtype=np.float32)
		for i in range(0,c,4):
			idx = i >> 2
			kernel[0,0,idx::out_c,idx]         = [1, 1, 1, 1]
			kernel[0,0,idx::out_c,idx+out_c]   = [1,-1, 1,-1]
			kernel[0,0,idx::out_c,idx+out_c*2] = [1, 1,-1,-1]
			kernel[0,0,idx::out_c,idx+out_c*3] = [1,-1,-1, 1]
		self.kernel = k.variable(value = kernel, dtype='float32')

	def call(self, inputs, **kwargs):
		y = k.conv2d(inputs, self.kernel, padding='same')
		y = tf.depth_to_space(y,2)
		return y

	def compute_output_shape(self, input_shape):
		c = input_shape[-1]>>2
		if(input_shape[1] != None and input_shape[2] != None):
			return (input_shape[0], input_shape[1] << 1, input_shape[2] << 1, c)
		else:
			return (None, None, None, c)

class TransitionLayer(layers.Layer):
	def __init__(self, ratio = 1, **kwargs):
		super(TransitionLayer, self).__init__(**kwargs)
		self.ratio = ratio
	
	def build(self, input_shape):
		c = input_shape[-1]
		self.output_c = 32#int(c*self.ratio)
		kernel_shape = (1,1,c,self.output_c)
		self.kernel = self.add_weight(
			shape = kernel_shape,
			initializer = initializers.get('he_normal'),
			regularizer=l2(1e-4),
			name = 'conv')
		self.built = True

	def call(self, inputs):
		y = k.conv2d(inputs,
			self.kernel,
			padding='same')
		return y
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1],input_shape[2],self.output_c)

class implicit_trans(layers.Layer):
	def __init__(self, q, **kwargs):
		self.q = q
		super(implicit_trans, self).__init__(**kwargs)

	def build(self, input_shape):
		conv_shape = (1,1,64,64)
		kernel = np.zeros(conv_shape)
		r1 = sqrt(1.0/8)
		r2 = sqrt(2.0/8)
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
		self.kernel = k.variable(value = kernel, dtype = 'float32')

	def call(self, inputs):
		y = k.conv2d(inputs,
					 self.kernel,
					 padding = 'same',
					 data_format='channels_last')
		return y

	def compute_output_shape(self, input_shape):
		return input_shape
	
class ScaleLayer(layers.Layer):
	def __init__(self, s, **kwargs):
		self.s = s
		super(ScaleLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(
			shape = (1,),
			name = 'scale',
			initializer=initializers.Constant(value=self.s))
	def call(self, inputs):
		return inputs*self.kernel

	def compute_output_shape(self, input_shape):
		return input_shape
