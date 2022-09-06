from json import detect_encoding
from Detector import * 
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
classFile = "class.names"
imagePath = "car.jpg"
threshold = 0.5

detector = Detector() 
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath, threshold)
