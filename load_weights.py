import numpy as np 


def Load_weights(model,weight_file):

	wf = open(weight_file, 'rb')
    #np.formfile(wf, dtype, count) : 讀取wf中的count個值，類似generator, but前五個值不是weight value
	major , minor, revision , seen, _ = np.fromfile(wf,dtype= np.int32, count=5)

	j=0
	# max of conv layer is 74
	for i in range(75):
        
		# conv layer namce is from 1-74
		conv_layer_name = 'conv2d_%d' %i if i>0 else 'conv2d'
		bn_layer_name = 'batch_normalization_%d' %j if j>0 else 'batch_normalization' 
        
		# use keras.model.get_layer 得到指定名稱的layers
		conv_layer = model.get_layer(conv_layer_name)
		#捲積核個數
		filters = conv_layer.filters
		#捲積核大小
		k_size = conv_layer.kernel_size[0]
		#conv layer的輸入形狀
		in_dim = conv_layer.input_shape[-1]

        # to get the weights of bn layers
		if i not in [58,66,74]: # layer 58, 66 and 74 dont have BN

			# darknet weights: [beta, gamma, mean, variance]
			bn_weights = np.fromfile(wf, dtype= np.float32, count = 4*filters)

			# tf weights: [gamma, beta, mean, variance]
			# reshape to tf weight
			bn_weights = bn_weights.reshape((4,filters))[[1,0,2,3]]
			bn_layer = model.get_layer(bn_layer_name)

			j+=1

		else:
			conv_bias = np.fromfile(wf,dtype= np.float32, count= filters)

		# darknet shape is (out_dim, in_dim, height,width)
		conv_shape = (filters, in_dim,k_size,k_size)
		conv_weights = np.fromfile(wf,dtype= np.float32, count= np.product(conv_shape))

		#tf shpae (height, width, in_dim, out_dim)
		conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])


		if i not in [58,66,74]:
			conv_layer.set_weights([conv_weights])
			bn_layer.set_weights(bn_weights)
		else:
			conv_layer.set_weights([conv_weights,conv_bias])

	assert len(wf.read(0))==0, 'failed to read all data'
	wf.close()

	return model




