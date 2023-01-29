import time
import numpy as np
import cv2
import tensorflow as tf

start = time.time()

image = cv2.imread('0rot-1trans250_250.jpg')
height, width, channel = image.shape
image_show = np.array(image, dtype='float32')
image_input = cv2.resize(image_show, (256,256))

TF_LITE_MODEL_FILE_NAME = "tf_lite_012rsmol_quant_model.tflite"
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image_input,0))
interpreter.invoke()
classification = interpreter.get_tensor(output_details[0]['index']).argmax()
regression = interpreter.get_tensor(output_details[1]['index'])
r = regression[0][0]
x = regression[0][1]
y = regression[0][2]

cv2.circle(image, (int(np.around(x*width+width/2)),int(np.around(y*height+height/2))), 190, (255,255,255), thickness=2)
cv2.putText(image, 'id'+str(classification) + ' ' + str(r*360) + ' degree', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 4, (255,255,255), thickness=2)
image = cv2.resize(image, (1000,750))
cv2.imshow('ta',image)

end = time.time()

print(str(end-start)+' seconds')
cv2.waitKey(0)
