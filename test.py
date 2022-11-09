import numpy as np
from keras_preprocessing import image
import cv2
import os
from keras.models import load_model
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
classes = ['Chuối', 'Dâu tây', 'Dứa', 'Khế', 'Măng cụt', 'Xoài']
new_model = load_model('model.h5')
from mss import mss 

def MainPredict():
    # r, frame = vid.read()

    # Test với 1 hình ảnh
    path = r"testdautay.jpg"
    # Reading an image in default mode
    frame = cv2.imread(path)

    cv2.imwrite(r'final' + str(i) + ".jpg", frame)
    test_image = image.load_img(
        r'final' + str(i) + ".jpg", target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)
    result1 = result[0]
    # Lấy độ dài của mảng
    len_classes = len(classes)
    print(len_classes)
    for y in range(len_classes):
        if result1[y] == 1.:
            prediction = classes[y]
            print(prediction)
            os.remove('final' + str(i) + ".jpg")
            cv2.imshow('frame', frame)
            #(this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0) 
            #closing all open windows 
            cv2.destroyAllWindows() 
            break
        # i = i + 1
        # if cv2.waitKey(1) & 0xFF == ord('q'):
MainPredict()
# vid.release()
# cv2.destroyAllWindows()
