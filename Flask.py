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


def cartoon (image):
	print(image)
	img = cv2.imread(image)	
	num_down = 3      	# number of downsampling steps
	num_bilateral = 18  # number of bilateral filtering steps
	img_rgb = img
	# downsample image using Gaussian pyramid
	
	img_color = img_rgb
	for _ in range(num_down):
		img_color = cv2.pyrDown(img_color)

	# repeatedly apply small bilateral filter instead of
	# applying one large filter
	for _ in range(num_bilateral):
		img_color = cv2.bilateralFilter(img_color, d=9,
										sigmaColor=9,
										sigmaSpace=7)

	# upsample image to original size
	for _ in range(num_down):
		img_color = cv2.pyrUp(img_color)

	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
	img_blur = cv2.medianBlur(img_gray, 5)

	img_edge = cv2.adaptiveThreshold(img_blur, 255,
									cv2.ADAPTIVE_THRESH_MEAN_C,
									cv2.THRESH_BINARY,
									blockSize=9,
									C=2)

	# convert back to color, bit-AND with color image
	img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
	img_cartoon = cv2.bitwise_and(img_color, img_edge)

	# display
	imS = cv2.resize(img_cartoon, (960, 540)) 
	# cv2.imshow("cartoon", imS)
	return img_cartoon

	#####################################################################################################################

def sketch(frame,param):	
	print(frame)
	img = cv2.imread(frame)
	# img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
	# img = cv2.resize(img, (160, 200)) 
	res , dst_color = cv2.pencilSketch(img, sigma_s=30, sigma_r=0.06, shade_factor=param)
	res = cv2.resize(res, (960, 540)) 
	# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	# res = cv2.filter2D(res, -1, sharpen_kernel)
	return res


	#####################################################################################################################



def sketchColor(frame,param):	
	print(frame)
	img = cv2.imread(frame)
	res , dst_color = cv2.pencilSketch(img, sigma_s=30, sigma_r=0.03, shade_factor=param)
	dst_color = cv2.resize(dst_color, (960, 540)) 
	
	return dst_color


	#####################################################################################################################

def oilPaint(image):
	img = cv2.imread(image)
	#first numeric argumet ko variable rakhna hai !! app mai ... jitna jyda utnaa jyda stroke hoga
	#second numeric argument ko variable rakhna hai !! app mai ... jinta jya uthna false counters aaengy 
	res = cv2.xphoto.oilPainting(img, 18, 5)
	res = cv2.resize(res, (960, 540)) 
	return res

	######################################################################################################################

def Detail(image):
	img = cv2.imread(image)
	res = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.6)
	res = cv2.resize(res, (760,540))
	return res


	######################################################################################################################

def EdgePreserving(image):
	img = cv2.imread(image)
	res = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
	res = cv2.resize(res,(760,540))
	return res


	######################################################################################################################

def glassFilter(filename):
	face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	filename='reconstructed.jpg'
	img=cv2.imread(filename)
	img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
	img = cv2.resize(img, (160, 200)) 
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print(gray)
	fl=face.detectMultiScale(gray)
	if(len(fl) == 0):
		return "could not detect the face !!"

	glass=cv2.imread('./FilterMask/glass1.png')
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
	t1 = cv2.getTickCount()
	for (x, y, w, h) in fl:
		frame = put_glass(glass, img, x, y, w, h)
	t2 = cv2.getTickCount()
	frame = cv2.resize(frame, (160, 200)) 
	sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
	frame = cv2.filter2D(frame, -1, sharpen_kernel)
	print("time taken = ",(t2-t1)/cv2.getTickFrequency())
	return frame


	######################################################################################################################


def catFilter(filename):
	face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	filename='reconstructed.jpg'
	img=cv2.imread(filename)
	img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
	img = cv2.resize(img, (160, 200)) 
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print(gray)
	fl=face.detectMultiScale(gray)
	dog=cv2.imread('./FilterMask/cat.png')
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
	frame = cv2.resize(frame, (160, 160)) 
	sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
	frame = cv2.filter2D(frame, -1, sharpen_kernel)
	print("time taken = ",(t2-t1)/cv2.getTickFrequency())
	return frame


	######################################################################################################################

def dog(filename):	
	face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	img=cv2.imread(filename)
	img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
	img = cv2.resize(img, (160, 200))
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	print(gray)
	fl=face.detectMultiScale(gray)
	dog=cv2.imread('./FilterMask/dog.png')
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
	if(len(fl) == 0):
		return "could not find a face"
	for (x, y, w, h) in fl:
		print("2'\n")
		frame = put_dog_filter(dog, img, x, y, w, h)
		print(frame)
	print("3\n")
	frame = cv2.resize(frame, (160, 200)) 
	sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
	frame = cv2.filter2D(frame, -1, sharpen_kernel)
	return frame


	######################################################################################################################


def Pointillism(image):
	parser = argparse.ArgumentParser(description='...')
	parser.add_argument('--palette-size', default=20, type=int, help="Number of colors of the base palette")
	parser.add_argument('--stroke-scale', default=10, type=int, help="Scale of the brush strokes (0 = automatic)")
	parser.add_argument('--gradient-smoothing-radius', default=2, type=int, help="Radius of the smooth filter applied to the gradient (0 = automatic)")
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


	######################################################################################################################
	######################################################################################################################
	######################################################################################################################
	######################################################################################################################
	######################################################################################################################


def HelperFunction(x):
	filename = "reconstructed.jpg"
	img = imread(io.BytesIO(base64.b64decode(x)))
	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	filename = "reconstructed.jpg"
	cv2.imwrite(filename, cv2_img)
	imS = cv2.resize(cv2_img, (960, 540)) 


@app.route('/watercolour',methods=['POST'])
def getDATAWatercolour():
	x= request.json['base64String']
	param = request.json['parameter']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = cartoon(filename,param)
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')

	######################################################################################################################

@app.route('/pencilsketchBW',methods = ['POST'])
def getDataPencilSketch():
	x=request.json['base64String']
	param = request.json['parameter']
	filename = "reconstructed.jpg"
	HelperFunction(x)
	opImg = sketch(filename,param)
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, op2_img)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')


######################################################################################################################


@app.route('/pencilsketchColor',methods = ['POST'])
def getDataPencilSketchColor():
	x=request.json['base64String']
	param = request.json['parameter']
	filename = "reconstructed.jpg"
	HelperFunction(x)
	opImg = sketchColor(filename,param)
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, op2_img)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')


######################################################################################################################


@app.route('/oilpaint',methods = ['POST'])
def getDataOilPaint():
	x=request.json['base64String']
	param = request.json['parameter']
	HelperFunction(x)
	filename = "reconstructed.jpg" 
	opImg = oilPaint(filename)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')

######################################################################################################################

@app.route('/pointlissim',methods = ['POST'])
def getDataPointLissim():
	x=request.json['base64String']
	param = request.json['parameter']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = Pointillism(filename)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')

######################################################################################################################

@app.route('/details',methods=['POST'])
def getDataDetails():
	x= request.json['base64String']
	param = request.json['parameter']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = Detail(filename)
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')

######################################################################################################################

@app.route('/edgePreserving',methods=['POST'])
def getDataEdgePreserve():
	x= request.json['base64String']
	param = request.json['parameter']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = EdgePreserving(filename)
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')

######################################################################################################################


@app.route('/dog',methods=['POST'])
def getDataDogFilter():
	x= request.json['base64String']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = dog(filename)
	if(opImg == "could not find a face"):
		return "could not find a face"
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')


######################################################################################################################


@app.route('/glass',methods=['POST'])
def getDataGlassFilter():
	x= request.json['base64String']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = glassFilter(filename)
	if(opImg == "could not find a face"):
		return "could not find a face"
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')


######################################################################################################################


@app.route('/cat',methods=['POST'])
def getDataCatFilter():
	x= request.json['base64String']
	HelperFunction(x)
	filename = "reconstructed.jpg"
	opImg = catFilter(filename)
	if(opImg == "could not find a face"):
		return "could not find a face"
	op2_img = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
	filename = "reconstructed1.jpg"
	cv2.imwrite(filename, opImg)
	return send_file(filename,as_attachment=True,mimetype='image/jpg')


######################################################################################################################

@app.errorhandler(404)
def page_not_found(e):
	filename = '../Assets/404_retry.jpg'
	return send_file(filename,as_attachment=True,mimetype='image/jpg')



@app.route('/ImageToText',methods=['POST'])
def ConvertImgToText():
	x=request.json['base64String']
	img = imread(io.BytesIO(base64.b64decode(x)))
	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	filename = "reconstructed.jpg"
	cv2.imwrite(filename, cv2_img)
	print(get_captcha_text_from_captcha_image(filename))
	return send_file(filename,as_attachment=True,mimetype='image/jpg') 

######################################################################################################################


# @app.route('/Objectdetection',methods = ['POST'])
# def DetectObject():
# 	x=request.json['base64String']
# 	img = imread(io.BytesIO(base64.b64decode(x)))
# 	cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 	filename = "reconstructed.jpg"
# 	cv2.imwrite(filename, cv2_img)



	
if __name__ == '__main__':
	app.run()