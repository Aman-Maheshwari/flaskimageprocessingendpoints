import cv2
import numpy as np

def GlassFilter():
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    filename='reconstructed.jpg'
    img=cv2.imread(filename)
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    img = cv2.resize(img, (160, 200)) 
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray)
    fl=face.detectMultiScale(gray)
    glass=cv2.imread('glass1.png')
    print("1")
    
    def put_glass(glass, fc, x, y, w, h):
        face_width = w
        face_height = h

        hat_width = face_width + 1
        hat_height = int(0.50 * face_height) + 1

        glass = cv2.resize(glass, (hat_width, hat_height))

        for i in range(hat_height):
            for j in range(hat_width):
                for k in range(3):
                    if glass[i][j][k] < 235:
                        fc[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
        return fc
    # print("fl = ",fl)
    t1 = cv2.getTickCount()
    for (x, y, w, h) in fl:
        print("2'\n")
        frame = put_glass(glass, img, x, y, w, h)
        print(frame)
    print("3\n")
    t2 = cv2.getTickCount()
    frame = cv2.resize(frame, (160, 200)) 
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    frame = cv2.filter2D(frame, -1, sharpen_kernel)
    print("time taken = ",(t2-t1)/cv2.getTickFrequency())
    return frame
