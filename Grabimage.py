import numpy as np
import PIL.ImageGrab
import cv2
import time
from  Keyboard import PressKey,ReleaseKey, W,A,S,D

def main():

    for i in list(range(4))[:: -1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()
    while(True):
   #     PressKey(W)
       # ReleaseKey(W)


        printscreen_numpy = np.array(PIL.ImageGrab.grab(bbox=(0,40,800,640)))
        print("loop take {} seconds".format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(printscreen_numpy)
        cv2.imshow("winndowTest",printscreen_numpy)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def process_img(image):
    original_image = image
    processed_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    process_img = cv2.Canny(processed_img,threshold1= 200,threshold2=300)
    return process_img


if __name__ == '__main__':
  main()