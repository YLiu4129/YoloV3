import tensorflow as tf 

# define BN
class BatchNormalization(tf.keras.layers.BatchNormalization):

	def call(self, x, training= False):

		if not training:
			training = tf.constant(False)
        
		# AND operator
		training = tf.logical_and(training, self.trainable)
        
		# return BN
		return super().call(x, training)




def convolutional(input_layer, filters_shape, down_sample = False,
		activate = True, batch_norm = True, regularization = 0.0005, 
		reg_stddev = 0.01, activate_alpha = 0.1):
    
	# if add strides then dont need padding
	if down_sample:
		input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)
		padding ="valid"
		strides = 2
	else:
		padding ="same"
		strides = 1
    
	# kernel_size = filters_shape[0] : (filters_shape[0], filters_shape[0])
	conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],
		kernel_size = filters_shape[0],
		strides = strides,
		padding = padding,
		use_bias = not batch_norm,
		kernel_regularizer= tf.keras.regularizers.l2(regularization),
		kernel_initializer = tf.random_normal_initializer(stddev=reg_stddev),
		bias_initializer = tf.constant_initializer(0.)
		)(input_layer)

	if batch_norm:
		conv = BatchNormalization()(conv)

	if activate:
		conv = tf.nn.leaky_relu(conv, alpha= activate_alpha)

	return conv

def res_block(input_layer, input_channel, filter_num1, filter_num2):

	#short cut layer
	short_cut = input_layer
	conv = convolutional(input_layer, filters_shape=(1,1,input_layer,filter_num1))
	conv = convolutional(conv, filters_shape=(3,3,filter_num1,filter_num2))

	res_output = short_cut+ conv 
	return res_output

'''the output consists of three vectors extracted by darknet53  the sizes of vectors are 13x13,
26x26, 52x52, 52x52 can learn small objects because 52x52 have more blocks, 13x13 learn big objects.
the third vector(botom layer of darknet53 will be input to Conv2D+upsampling layer which is added
to second vector, and at the same
time, it will be input to 3x3 conv2D filter and 1x1 conv2d filter to be adjusted to
(batch, 13, 13, anchor_num, x, y, height, width, confidence rate, classes)
which mean every anchor box has (x_offset, y_offset, height, width, confidence rate, classes)
notice: every vector has 3 different sizes of anchor boxes, so total 9 anchor boxes'''



def darknet53(input_data):
	input_data = convolutional(input_data,(3,3,3,32))
	input_data = convolutional(input_data, (3,3,32,64), down_sample = True)

	for i in range(1):
		input_data = res_block(input_data, 64,32,64)

	input_data = convolutional(input_data, (3,3,64,128),down_sample=True)

	for i in range(2):
		input_data = res_block(input_data, 128,64,128)

	input_data = convolutional(input_data, (3,3,128,256), down_sample= True)

	for i in range(8):
		input_data = res_block(input_data,256,128,256)

	vec_1 = input_data 

	input_data = convolutional(input_data,(3,3,256,512), down_sample= True)

	for i in range(8):
		input_data = res_block(input_data,512,256,512)

	vec_2 = input_data

	input_data = convolutional(input_data,(3,3,512,1024), down_sample= True)

	for i in range(4):
		input_data = res_block(input_data,1024,512,1024)

	vec_3 = input_data

	return vec_1, vec_2, vec_3

def upsample(input_layer):
    
	#tf.image.resize_images(image， （w， h）， method)
	return tf.image.resize(input_layer,(input_layer.shape[1]*2,input_layer.shape[2]*2),
		method='nearest')

