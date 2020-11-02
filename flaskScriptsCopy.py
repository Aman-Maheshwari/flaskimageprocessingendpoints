# import numpy as np
# from flask import Flask, jsonify, request,send_from_directory,send_file
# import cv2
# import json
# import io
# from imageio import imread
# import base64
# import matplotlib.pyplot as plt
# import sys

# app =Flask(__name__)
# app.config["DEBUG"] = True

# highThresh	= 0.4
# lowThresh		= 0.1
# # imgFileLst	= ('./villageRoad.jpg', './me.jpg','./opencvframe0.png')
# def sobel (img):
# 	'''
# 	Detects edges using sobel kernel
# 	'''
# 	opImgx		= cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)	#detects horizontal edges
# 	opImgy		= cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)	#detects vertical edges
# 	#combine both edges
# 	return cv2.bitwise_or(opImgx,opImgy)	#does a bitwise OR of pixel values at each pixel
# def sketch(frame):	
# 	#Blur it to remove noise
# 	frame		= cv2.GaussianBlur(frame,(6,6),cv2.BORDER_DEFAULT)
	
# 	#make a negative image
# 	invImg	= 255-frame
	
# 	#Detect edges from the input image and its negative
# 	edgImg0		= sobel(frame)
# 	edgImg1		= sobel(invImg)
# 	edgImg		= cv2.addWeighted(edgImg0,1,edgImg1,1,0)	#different weights can be tried too
	
# 	#Invert the image back
# 	opImg= 255-edgImg
# 	# imS = cv2.resize(opImg, (960, 540)) 
# 	# cv2.imshow ('imagee',imS)
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()
# 	return opImg
# @app.route('/',methods=['POST'])
# def getDATA():
# 	x=request.json['base64String']
# 	img = imread(io.BytesIO(base64.b64decode(x)))
# 	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 	filename = "reconstructed.jpg"
# 	cv2.imwrite(filename, cv2_img)
# 	# imS = cv2.resize(cv2_img, (960, 540)) 
# 	# cv2.imshow ('imagee',imS)
# 	# plt.show()
# 	# cv2.waitKey(0)
# 	# cv2.destroyAllWindows()

# 	opImg = sketch(img)
# 	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
# 	filename = "reconstructed1.jpg"
# 	cv2.imwrite(filename, op2_img)

# 	# cv2.imshow ('',opImg)
# 	# b64_bytes = base64.b64encode(opImg)
# 	# b64_string = b64_bytes.decode()
# 	# print(sys.getsizeof(b64_string))
# 	# print(b64_string)
# 	# print(x)
# 	return send_file(filename,as_attachment=True,mimetype='image/jpg')
# 	# return json.dumps({'file_id': b64_string ,'filename': "record.filename" , 'links_to' : "record.links_to"})
# if __name__ == '__main__':
# 	app.run()
# 	# for imgFile in imgFileLst:
# 	# 	print (imgFile)
# 	# 	img		= cv2.imread (imgFile,0)
# 	# 	opImg	= sketch(img)	
# 	# 	cv2.imshow (imgFile,opImg)
	

import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask import Flask, jsonify, request,send_from_directory,send_file
import json
import io
from imageio import imread
import base64
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageFilter, ImageChops
from pytesseract import image_to_string
import pytesseract
import numpy
import argparse
import math
import progressbar
from pointillism import *

app =Flask(__name__)
app.config["DEBUG"] = True
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image_using_pil(image_path):
    # unblur, sharpen filters
    img = Image.open(image_path)
    img = img.convert("RGBA")

    pixdata = img.load()

    # Make the letters bolder for easier recognition
    
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][0] < 90:
                pixdata[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][1] < 136:
                pixdata[x, y] = (0, 0, 0, 255)

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][2] > 0:
                pixdata[x, y] = (255, 255, 255, 255)

    # And sharpen it
    img.filter(ImageFilter.SHARPEN)
    img.save("input-black.gif")

    #   Make the image bigger (needed for OCR)
    basewidth = 1000  # in pixels
    im_orig = Image.open('input-black.gif')
    wpercent = (basewidth/float(im_orig.size[0]))
    hsize = int((float(im_orig.size[1])*float(wpercent)))
    big = img.resize((basewidth, hsize), Image.ANTIALIAS)

    # tesseract-ocr only works with TIF so save the bigger image in that format
    ext = ".tif"
    tif_file = "input-NEAREST.tif"
    big.save(tif_file)
    
    return tif_file


def get_captcha_text_from_captcha_image(captcha_path):
	# Preprocess the image befor OCR
	tif_file = preprocess_image_using_pil(captcha_path)
    # Perform OCR using tesseract-ocr library
	image = Image.open(tif_file)
	ocr_text = image_to_string(image)
	alphanumeric_text = ''.join(e for e in ocr_text)
	return alphanumeric_text



# def FindObject(img):
# 	img = cv2.imread(img,cv2.IMREAD_COLOR)
# 	img1=img
# 	grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 	th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
# 	kernel = np.ones((5,5),np.float32)
# 	opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
# 	blur=cv2.bilateralFilter(opening,19,75,75)

# 	dilate=cv2.dilate(blur,kernel,iterations=1)

# 	cv2.namedWindow('Blurring',cv2.WINDOW_NORMAL)
# 	cv2.resizeWindow('Blurring', 800,600)

# 	edged = cv2.Canny(dilate, 50, 150) #canny edge detection
# 	cv2.namedWindow('Canny',cv2.WINDOW_NORMAL)
# 	cv2.resizeWindow('Canny', 800,600)
# 	cv2.imshow('Canny',edged)


# 	cv2.imshow('Blurring',dilate)

# 	contours, hierarchy = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 	for contour in contours:
# 		epsilon = 0.01*cv2.arcLength(contour,True)
# 		approx = cv2.approxPolyDP(contour,epsilon,True)
# 		cv2.drawContours(img1,contour,-1,(255,0,0),4)
# 	cv2.namedWindow('final',cv2.WINDOW_NORMAL)
# 	cv2.resizeWindow('final', 800,600)
# 	# cv2.imshow('final',img1)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()


def cartoon (image):
	# import cv2
	print(image)
	img = cv2.imread(image)
	res , dst_color = cv2.pencilSketch(img, sigma_s=30, sigma_r=0.05, shade_factor=0.02)
	# res = cv2.xphoto.oilPainting(img, 38, 1)
	res = cv2.resize(res, (960, 540)) 
	cv2.imshow("Frame",res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return res
	# num_down = 3      # number of downsampling steps
	# num_bilateral = 18  # number of bilateral filtering steps

	# img_rgb = image

	# # downsample image using Gaussian pyramid
	# img_color = img_rgb
	# for _ in range(num_down):
	# 	img_color = cv2.pyrDown(img_color)

	# # repeatedly apply small bilateral filter instead of
	# # applying one large filter
	# for _ in range(num_bilateral):
	# 	img_color = cv2.bilateralFilter(img_color, d=9,
	# 									sigmaColor=9,
	# 									sigmaSpace=7)

	# # upsample image to original size
	# for _ in range(num_down):
	# 	img_color = cv2.pyrUp(img_color)

	# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	# img_blur = cv2.medianBlur(img_gray, 5)

	# img_edge = cv2.adaptiveThreshold(img_blur, 255,
	# 								cv2.ADAPTIVE_THRESH_MEAN_C,
	# 								cv2.THRESH_BINARY,
	# 								blockSize=9,
	# 								C=2)

	# 								# convert back to color, bit-AND with color image
	# img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
	# img_cartoon = cv2.bitwise_and(img_color, img_edge)

	# # display
	# imS = cv2.resize(img_cartoon, (960, 540)) 
	# cv2.imshow("cartoon", imS)
	# return img_cartoon


highThresh	= 0.4
lowThresh		= 0.1
def sobel (img):
	'''
	Detects edges using sobel kernel
	'''
	opImgx		= cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)	#detects horizontal edges
	opImgy		= cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)	#detects vertical edges
	#combine both edges
	return cv2.bitwise_or(opImgx,opImgy)	#does a bitwise OR of pixel values at each pixel
def sketch(frame):	
	#Blur it to remove noise
	frame		= cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
	
	#make a negative image
	invImg	= 255-frame
	
	#Detect edges from the input image and its negative
	edgImg0		= sobel(frame)
	edgImg1		= sobel(invImg)
	edgImg		= cv2.addWeighted(edgImg0,1,edgImg1,1,0)	#different weights can be tried too
	
	#Invert the image back
	opImg= 255-edgImg
	# imS = cv2.resize(opImg, (960, 540)) 
	# cv2.imshow ('imagee',imS)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return opImg


def dog(filename):	
	face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	img=cv2.imread(filename)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print(gray)
	fl=face.detectMultiScale(gray)
	dog=cv2.imread('dog.png')
	print("1")
	def put_dog_filter(dog, fc, x, y, w, h):
		print("2\n")
		face_width = w
		face_height = h

		dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
		for i in range(int(face_height * 1.75)):
			for j in range(int(face_width * 1.5)):
				for k in range(3):
					if dog[i][j][k] < 235:
						fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
		return fc
	print("fl = ",fl)
	for (x, y, w, h) in fl:
		print("2'\n")
		frame = put_dog_filter(dog, img, x, y, w, h)
		print(frame)
	print("3\n")
	frame = cv2.resize(frame, (160, 200)) 
	cv2.imshow('image',frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return frame


def Pointillism(image):
	parser = argparse.ArgumentParser(description='...')
	parser.add_argument('--palette-size', default=20, type=int, help="Number of colors of the base palette")
	parser.add_argument('--stroke-scale', default=0, type=int, help="Scale of the brush strokes (0 = automatic)")
	parser.add_argument('--gradient-smoothing-radius', default=0, type=int, help="Radius of the smooth filter applied to the gradient (0 = automatic)")
	parser.add_argument('--limit-image-size', default=0, type=int, help="Limit the image size (0 = no limits)")
	parser.add_argument('img_path', nargs='?', default=image)

	args = parser.parse_args()

	res_path = args.img_path.rsplit(".", -1)[0] + "_drawing.jpg"
	img = cv2.imread(args.img_path)

	if args.limit_image_size > 0:
		img = limit_size(img, args.limit_image_size)

	if args.stroke_scale == 0:
		stroke_scale = int(math.ceil(max(img.shape) / 1000))
		print("Automatically chosen stroke scale: %d" % stroke_scale)
	else:
		stroke_scale = args.stroke_scale

	if args.gradient_smoothing_radius == 0:
		gradient_smoothing_radius = int(round(max(img.shape) / 50))
		print("Automatically chosen gradient smoothing radius: %d" % gradient_smoothing_radius)
	else:
		gradient_smoothing_radius = args.gradient_smoothing_radius

	# convert the image to grayscale to compute the gradient
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	print("Computing color palette...")
	palette = ColorPalette.from_image(img, args.palette_size)

	print("Extending color palette...")
	palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

	# display the color palette
	cv2.imshow("palette", palette.to_image())
	cv2.waitKey(200)

	print("Computing gradient...")
	gradient = VectorField.from_gradient(gray)

	print("Smoothing gradient...")
	gradient.smooth(gradient_smoothing_radius)

	print("Drawing image...")
	# create a "cartonized" version of the image to use as a base for the painting
	res = cv2.medianBlur(img, 11)
	# define a randomized grid of locations for the brush strokes
	grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
	batch_size = 10000

	bar = progressbar.ProgressBar()
	for h in bar(range(0, len(grid), batch_size)):
		# get the pixel colors at each point of the grid
		pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
		# precompute the probabilities for each color in the palette
		# lower values of k means more randomnes
		color_probabilities = compute_color_probabilities(pixels, palette, k=9)

		for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
			color = color_select(color_probabilities[i], palette)
			angle = math.degrees(gradient.direction(y, x)) + 90
			length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))

			# draw the brush stroke
			cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)


	cv2.imshow("res", limit_size(res, 1080))
	# cv2.imwrite(res_path, res)
	cv2.waitKey(0)
	return res

# @app.route('/',methods=['POST'])
# def getDATA():
# 	x= request.json['base64String']
# 	height = request.json['height']
# 	width = request.json['width']
# 	filename = "reconstructed.jpg"
# 	img = imread(io.BytesIO(base64.b64decode(x)))
# 	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 	filename = "reconstructed.jpg"
# 	cv2.imwrite(filename, cv2_img)
# 	imS = cv2.resize(cv2_img, (width, height)) 
# 	cv2.imshow ('imagee',imS)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()
# 	opImg = cartoon(filename)
# 	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
# 	filename = "reconstructed1.jpg"
# 	cv2.imwrite(filename, opImg)
# 	cv2.imshow ('',opImg)
# 	return send_file(filename,as_attachment=True,mimetype='image/jpg')


@app.route('/watercolour',methods=['POST'])
def getDATA():
	x= request.json['base64String']
	height = request.json['height']
	width = request.json['width']
	filename = "reconstructed.jpg"
	img = imread(io.BytesIO(base64.b64decode(x)))
	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	filename = "reconstructed.jpg"
	cv2.imwrite(filename, cv2_img)
	imS = cv2.resize(cv2_img, (width, height)) 
	cv2.imshow ('imagee',imS)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	opImg = cartoon(filename)
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	cv2.imshow ('',opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')

@app.route('/pencil',methods = ['POST'])
def PencilSketch():
		x=request.json['base64String']
		img = imread(io.BytesIO(base64.b64decode(x)))
		cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		filename = "reconstructed.jpg"
		cv2.imwrite(filename, cv2_img)
		imS = cv2.resize(cv2_img, (960, 540)) 
		cv2.imshow ('imagee',imS)
		# plt.show()
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		opImg = sketch(img)
		op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
		filename = "reconstructed1.jpg"
		cv2.imwrite(filename, op2_img)

		# cv2.imshow ('',opImg)
		# b64_bytes = base64.b64encode(opImg)
		# b64_string = b64_bytes.decode()
		# print(sys.getsizeof(b64_string))
		# print(b64_string)
		# print(x)
		return send_file(filename,as_attachment=True,mimetype='image/jpg')
		# return json.dumps({'file_id': b64_string ,'filename': "record.filename" , 'links_to' : "record.links_to"})


@app.route('/ImageToText',methods=['POST'])
def ConvertImgToText():
	x=request.json['base64String']
	img = imread(io.BytesIO(base64.b64decode(x)))
	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	filename = "reconstructed.jpg"
	cv2.imwrite(filename, cv2_img)
	# imS = cv2.resize(cv2_img, (960, 540)) 
	# cv2.imshow ('imagee',imS)
	print(get_captcha_text_from_captcha_image(filename))
	return send_file(filename,as_attachment=True,mimetype='image/jpg') 


# @app.route('/Objectdetection',methods = ['POST'])
# def DetectObject():
# 	x=request.json['base64String']
# 	img = imread(io.BytesIO(base64.b64decode(x)))
# 	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 	filename = "reconstructed.jpg"
# 	cv2.imwrite(filename, cv2_img)



	
if __name__ == '__main__':
	app.run()