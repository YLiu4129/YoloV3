import tensorflow as tf 
from commonBlocks import darknet53,upsample,convolutional
import numpy as np
from load_weights import Load_weights
 

# hyperparameters 
NUM_CLASSES = 80

# 三個輸出向量的縮放值 分別對應(52,26,13)
STRIDES = np.array([8,16,32])

# anchor框大小
ANCHORS =(1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 
          3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875)
ANCHORS = np.array(ANCHORS).reshape(3,3,2)
weight_file = "./weights/yolov3.weights"


def yoloV3(input_layer):

	route_1, route_2, conv = darknet53(input_layer) 
    
	# 5 times of conv
	# with 1x1 and 3x3 filters
	conv = convolutional(conv, (1,1,1024,512)) 
	conv = convolutional(conv,(3,3,512,1024)) 
	conv = convolutional(conv, (1,1,1024, 512))  
	conv = convolutional(conv, (3,3,512,1024))  
	conv = convolutional(conv,(1,1,1024,512)) 
    
	# extract 13x13 vector for output
	# conv_large_object_branch
	conv_lobj_branch = convolutional(conv,(3,3,512,1024))
	conv_lbbox = convolutional(conv_lobj_branch,(1,1,1024,3*(NUM_CLASSES+5)),
		activate= False, batch_norm = False)
    
	#conv+upsample for feature integration
	conv = convolutional(conv,(1,1,512,256))
	conv = upsample(conv)
    
	# concat with vec_2
	conv = tf.concat([conv, route_2], axis =-1) 
	conv = convolutional(conv,(1,1,768,256)) 
	conv = convolutional(conv,(3,3,256, 512))
	conv = convolutional(conv,(1,1,512,256))
	conv = convolutional(conv,(3,3,256,512))
	conv = convolutional(conv, (1,1,512,256))
    
	#extract vec_2
	conv_mobj_branch = convolutional(conv, (3,3,256,512))
	conv_mbbox = convolutional(conv_mobj_branch ,(1,1,512,3*(NUM_CLASSES+5)),
		activate= False, batch_norm= False)

    
	conv = convolutional(conv, (1,1,256,128))
	conv = upsample(conv)

	conv = tf.concat([conv,route_1], axis = -1)

	conv = convolutional(conv, (1,1,384,128))
	conv = convolutional(conv, (3,3,128, 256))
	conv = convolutional(conv, (1,1,256, 128))
	conv = convolutional(conv, (3,3,128, 256))
	conv = convolutional(conv, (1,1,256, 128))
    
	# extract vec_1 
	conv_sobj_branch = convolutional(conv,(3,3,128, 256))
	conv_sbbox = convolutional(conv_sobj_branch,
		(1,1,256,3*(NUM_CLASSES+5)),activate= False , batch_norm= False)

	return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_out, i = 0):

	conv_shape = tf.shape(conv_out)
	batch_size = conv_shape[0]
	# 13, 26, 52
	output_size = conv_shape[1]
    
	# shape :　(batch size, output_size, outputsize, 3, 5+num_classes)
	conv_output = tf.reshape(conv_out, (batch_size, output_size,output_size, 3,5+NUM_CLASSES))
	
	#[:,:,:,:,0:2] : [batch, 13/26/52, 13/26/52, anchor nums, dx and dy]
	#模型預測的dx dy dw dh為grid 中心點的座標，相當於label box x, y, w, h的偏移量
	conv_raw_dxdy = conv_output[:,:,:,:,0:2]
	# get dw dh
	conv_raw_dwdh = conv_output[:,:,:,:,2:4]
	#confidence score
	conv_raw_conf = conv_output[:,:,:,:,4:5]
	#class probability
	conv_raw_prob = conv_output[:,:,:,:,5:]
    
	# 構建一個 y 軸方向 (conv_bbox_size, conv_bbox_size) 大小的tensor,
    # 并填入对应的正数值，用来表示它的绝对位置
	# eg:tf.range(output_size,dtype=tf.int32)[:,tf.newaxis] -> (13, 1)
	# eg:tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size]) -> (13,13)
	# tf.tile((x,y)): 先對第一維擴展x倍, 再對第二維擴展y倍
	y = tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size])
	x = tf.tile(tf.range(output_size, dtype= tf.int32)[tf.newaxis,:],[output_size,1])
    
    # 將 (conv_bbox_size, conv_bbox_size) 大小的tensor concat
    # 得到 (conv_bbox_size, conv_bbox_size, 2) 大小的tensor, 就得到對應的 feature map 每個grid絕對位置的數值
	# eg:(0,0),(0,1)~(0,12)
	xy_grid = tf.concat([x[:,:,tf.newaxis],y[:,:,tf.newaxis]], axis = -1)

	# 利用tf.tile構建成 (batch_size, conv_bbox_size, conv_bbox_size, anchor_per_scale, 2) 
	xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
	xy_grid = tf.cast(xy_grid,tf.float32)
    
	# get x and y 's prediction value through sigmoid and map to the center of original image
	#  (偏移量 + 左上角坐标值) * 缩放值
	pred_xy = (tf.sigmoid(conv_raw_dxdy)+xy_grid)*STRIDES[i]

	# 获取 w、h 预测值 映射到 原图 的 width 和 high
    # equation : b_w = p_w * e ^ (t_w) and mutiply rescale value to get coordinate of original image
    # p_w is the w size of anchor box 
	pred_wh = (tf.exp(conv_raw_dwdh)*ANCHORS[i])*STRIDES[i]

	# concatenate all 
	pred_xywh = tf.concat([pred_xy,pred_wh], axis = -1)
    
	# prediction of confidence score and class probability
	pred_conf = tf.sigmoid(conv_raw_conf)
	pred_prob = tf.sigmoid(conv_raw_prob)
    
	# return [batch_size, conv_bbox_size, conv_bbox_size, anchor_per_scale, 4 + 1 + class_num] 
    # 4 + 1 + class_num : pred_xywh + pred_conf + pred_prob
    
	return tf.concat([pred_xywh, pred_conf, pred_prob], axis = -1)

def Model():
	input_layer = tf.keras.layers.Input([416,416,3])
	feature_maps = yoloV3(input_layer)

	bbox_tensors = []
    
	# decode every vector
	for i , fm in enumerate(feature_maps):
		bbox_tensor = decode(fm, i)
		bbox_tensors.append(bbox_tensor)

    # call keras model
	model = tf.keras.Model(input_layer, bbox_tensors)

	# load weight
	model = Load_weights(model, weight_file)

	return model 








