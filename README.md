# YoloV3
YoloV3 object detection algorithm using Tensorflow 2.1.0

### Prerequisites
Tensorflow 2.1.0 <br>
Numpy <br>
PILLOW (Python image processing library) <br>
OpenCV <br>

or use code

`
pip3 install -r ./requirements.txt` <br>

`wget https://pjreddie.com/media/files/yolov3.weights -O ./yolov3.weights`

## How to use

1.Please download coco weights from https://pjreddie.com/media/files/yolov3.weights and put it in 'weights' file. <br>

2.Put your image/video file in 'data' file. <br>

3.Go to the file path of detect.py in terminal and type: <br>

`python detect.py --mode image --data_dir <your file path>` if you use image or <br>

`python detect.py --mode video --data_dir <your file path>` <br>

4.You can check detect.py file do set the hyperparameter

## Reference
1.https://pjreddie.com/media/files/papers/YOLOv3.pdf <br>
2.https://pjreddie.com/darknet/yolo/ <br>
3.https://github.com/zzh8829/yolov3-tf2 <br>

