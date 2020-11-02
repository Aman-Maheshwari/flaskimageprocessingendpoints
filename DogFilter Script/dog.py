# # import cv2

# # face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# # filename='gs.jpg'
# # img=cv2.imread('./gs.jpg')
# # # cv2.imshow('image',img)
# # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # fl=face.detectMultiScale(gray,1.09,7)
# # dog=cv2.imread('./dog.png')


# # def put_dog_filter(dog, fc, x, y, w, h):
# #     face_width = w
# #     face_height = h
# #     dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
# #     for i in range(int(face_height * 1.75)):
# #         for j in range(int(face_width * 1.5)):
# #             for k in range(3):
# #                 if dog[i][j][k] < 235:
# #                     fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
# #     return fc

# # for (x, y, w, h) in fl:
# #     f=put_dog_filter(dog, img, x, y, w, h)

# # cv2.imshow('img',f)
# # cv2.waitKey(20000)& 0xff
# # cv2.destroyAllWindows()

# import cv2
# import numpy as np

# face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# filename='reconstructed.jpg'
# img=cv2.imread(filename)
# img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
# img = cv2.resize(img, (160, 200)) 
# # cv2.imshow('image',img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(gray)
# fl=face.detectMultiScale(gray)
# glass=cv2.imread('hallowene.png')
# print("1")

# # def put_dog_filter(dog, fc, x, y, w, h):
# #     print("2\n")
# #     face_width = w
# #     face_height = h

# #     dog = cv2.resize(dog, (int(face_width * 1.65), int(face_height * 1.1)))
# #     for i in range(int(face_height * 1.1)):
# #         for j in range(int(face_width * 1.65)):
# #             for k in range(3):
# #                 if dog[i][j][k] < 235:
# #                     fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
# #     return fc

# def put_glass(glass, fc, x, y, w, h):
#     face_width = w
#     face_height = h

#     hat_width = face_width+20
#     hat_height = int(0.10 * face_height)+50

#     glass = cv2.resize(glass, (hat_width, hat_height))

#     for i in range(hat_height):
#         for j in range(hat_width):
#             for k in range(3):
#                 if glass[i][j][k] < 235:
#                     fc[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
#     return fc
# # print("fl = ",fl)
# t1 = cv2.getTickCount()
# for (x, y, w, h) in fl:
#     print("2'\n")
#     frame = put_glass(glass, img, x, y, w, h)
#     print(frame)
# print("3\n")
# t2 = cv2.getTickCount()
# frame = cv2.resize(frame, (160, 200)) 
# sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
# frame = cv2.filter2D(frame, -1, sharpen_kernel)
# cv2.imshow('image',frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("time taken = ",(t2-t1)/cv2.getTickFrequency())



import cv2
import numpy as np 


#get facial classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#read images
witch = cv2.imread('witchHat.png')

#get shape of witch
original_witch_h,original_witch_w,witch_channels = witch.shape

#convert to gray
witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

#create mask and inverse mask of witch
ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#read video
img = cv2.imread('reconstructed.jpg')
img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
img = cv2.resize(img, (160, 200)) 
img_h, img_w = img.shape[:2]
    
#read each frame of video and convert to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#find faces in image using classifier
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#for every face found:
for (x,y,w,h) in faces:
    #retangle for testing purposes
    #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #coordinates of face region
    face_w = w
    face_h = h
    face_x1 = x
    face_x2 = face_x1 + face_w
    face_y1 = y
    face_y2 = face_y1 + face_h

    #witch size in relation to face by scaling
    witch_width = int(1.5 * face_w)
    witch_height = int(witch_width * original_witch_h / original_witch_w)
    
    #setting location of coordinates of witch
    witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
    witch_x2 = witch_x1 + witch_width
    witch_y1 = face_y1 - int(face_h*1.25)
    witch_y2 = witch_y1 + witch_height 

    #check to see if out of frame
    if witch_x1 < 0:
        witch_x1 = 0
    if witch_y1 < 0:
        witch_y1 = 0
    if witch_x2 > img_w:
        witch_x2 = img_w
    if witch_y2 > img_h:
        witch_y2 = img_h

    #Account for any out of frame changes
    witch_width = witch_x2 - witch_x1
    witch_height = witch_y2 - witch_y1

    #resize witch to fit on face
    witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

    #take ROI for witch from background that is equal to size of witch image
    roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

    #original image in background (bg) where witch is not
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
    roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
    dst = cv2.add(roi_bg,roi_fg)

    #put back in original image
    img[witch_y1:witch_y2, witch_x1:witch_x2] = dst
    #display image
img = cv2.resize(img, (460, 400)) 
cv2.imshow('img',img) 
cv2.waitKey(0)
cv2.destroyAllWindows() #close all windows