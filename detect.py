from model import Model
import cv2 
from absl import logging
import time
import numpy
import tensorflow as tf
import utils
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='image', type=str, help='img or video detection')
    parser.add_argument('--num_classes', default=80, type=int, help='nums of classes you want to detection')
    parser.add_argument('--data_dir', default='D:\coding\Projects\YoloV3\YOLO_V3\data',
                        type=str, help='the directory of the data')
    args = parser.parse_args()

    # file of video
    if not args.data_dir :
        print('Please input data directory...')
    if args.mode is None:
        print('Please input data type. Image or video?')
    file_lst = os.listdir(args.data_dir)
    if file_lst:
        image_files = []
        for file in file_lst:
            if '.jpg' in file or '.png' in file and args.mode == 'image':
                img_path = args.data_dir + '/' + file
                image_files.append(img_path)
            else :
                videoFile = args.data_dir + '/' + file
                video_out = "converted.mp4"
    else:
        print('Error : Empty file')
    #print(f'file name: {image_files}')
    print(args)

    mode = args.mode

    def process(mode):

        model = Model()
        names = utils.read_class_names("classes.names")

        if args.mode == "video" or args.mode == "camera":
            print('Start decode video...')
            #load video
            if args.mode == "video":
                # open video file
                vid = cv2.VideoCapture(videoFile)
            elif args.mode =="camera":
                vid = cv2.VideoCapture(0)

            # 取得影像的尺寸大小
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # get fps
            fps = int(vid.get(cv2.CAP_PROP_FPS))

            #建立視訊流寫入物件，VideoWriter_fourcc為視訊編解碼器，20為幀播放速率
            # *XVID: MPEG-4编码
            codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            output = cv2.VideoWriter(video_out, codec, fps, (width, height), 0)
            start = True
            print('Start YoloV3 detection...')
            while start:
                # get imgs

                ret, img = vid.read()

                if img is None:
                    logging.warning("Empty Frame")
                    start = False
                    time.sleep(0.1)
                    continue


                img_size = img.shape[:2]

                img_in = tf.expand_dims(img, 0)
                #make image size to (416,416)
                img_in = utils.transform_images(img_in, 416)
                # input image to model to get predict box
                # pred_bbox is a list containing three vectors
                # pred_bbox shape：(1, 52, 52, 3, 15)，(1, 26, 26, 3, 15)，(1, 13, 13, 3, 15)
                # change the shape to (3, -1, 85)
                pred_bbox = model.predict(img_in)
                pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                #(-1, 85)
                pred_bbox = tf.concat(pred_bbox, axis=0)

                boxes ,class_names, scores=utils.box_detector(pred_bbox, args.num_classes)
                img = utils.drawbox(boxes ,class_names, scores,names,img)

                if video_out:
                    output.write(img)
                img = cv2.resize(img, (900, 600))
                cv2.imshow('output', img)


                if cv2.waitKey(1) == ord('q'):

                    break


            vid.release()
            #output.release()
            cv2.destroyAllWindows()

            #if 'tf_yolov3.h5' not in os.listdir('D:\coding\Projects\YoloV3\YOLO_V3'):
                #print('Saving model...')
                #model.save('tf_yolov3.h5')

            print('Completed...')

        if args.mode == "image":

            for i, image_file in enumerate(image_files):
                img = cv2.imread(image_file)
                img_in = tf.expand_dims(img,0)

                img_in = utils.transform_images(img_in,416)
                pred_bboxes = model.predict(img_in)
                #print(numpy.array(pred_bboxes[0]).shape)
                # input image to model to get predict box
                # pred_bbox is a list containing three vectors
                # pred_bbox shape：(1, 52, 52, 3, 85)，(1, 26, 26, 3, 85)，(1, 13, 13, 3, 85)
                # change the shape to (-1,85)
                pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]
                #print(numpy.array(pred_bbox[0]).shape)
                # concatenate all pred boxes
                pred_bbox = tf.concat(pred_bbox, axis=0)
                #print(pred_bbox.shape)
                boxes ,class_names, scores=utils.box_detector(pred_bbox, args.num_classes)
                #print(boxes.shape)
                img = utils.drawbox(boxes ,class_names, scores,names,img)

                #img = cv2.resize(img, (800, 800))
                cv2.imwrite("output_%d.jpg"%i,img)
                cv2.imshow('output', img)

                if cv2.waitKey(0) == ord('q'):
                    cv2.destroyAllWindows()

                #if 'tf_yolov3.h5' not in os.listdir('D:\coding\Projects\YoloV3\YOLO_V3'):
                    #print('Saving model...')
                    #model.save('tf_yolov3.h5')

                print('Completed...')


    if args.mode == 'video':
            process(args.mode)
    else:
            args.mode = 'image'
            process(args.mode)
