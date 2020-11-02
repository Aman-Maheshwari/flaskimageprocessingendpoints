# import cv2

# face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# filename='gs.jpg'
# img=cv2.imread('./gs.jpg')
# # cv2.imshow('image',img)
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# fl=face.detectMultiScale(gray,1.09,7)
# dog=cv2.imread('./dog.png')


# def put_dog_filter(dog, fc, x, y, w, h):
#     face_width = w
#     face_height = h
#     dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
#     for i in range(int(face_height * 1.75)):
#         for j in range(int(face_width * 1.5)):
#             for k in range(3):
#                 if dog[i][j][k] < 235:
#                     fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
#     return fc

# for (x, y, w, h) in fl:
#     f=put_dog_filter(dog, img, x, y, w, h)

# cv2.imshow('img',f)
# cv2.waitKey(20000)& 0xff
# cv2.destroyAllWindows()

import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
filename='reconstructed.jpg'
img=cv2.imread(filename)
img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
img = cv2.resize(img, (160, 200)) 
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray)
fl=face.detectMultiScale(gray)
dog=cv2.imread('cat.png')
print("1")

def put_dog_filter(dog, fc, x, y, w, h):
    print("2\n")
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.65), int(face_height * 1.1)))
    for i in range(int(face_height * 1.1)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
    return fc
# print("fl = ",fl)
t1 = cv2.getTickCount()
for (x, y, w, h) in fl:
    print("2'\n")
    frame = put_dog_filter(dog, img, x, y, w, h)
    print(frame)
print("3\n")
t2 = cv2.getTickCount()
frame = cv2.resize(frame, (160, 200)) 
cv2.imshow('image',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("time taken = ",(t2-t1)/cv2.getTickFrequency())
