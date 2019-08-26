import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from operator import itemgetter
from PIL import Image
import random 
import sys
import os, sys
import tensorflow.compat.v1 as tf
from astropy.wcs import WCS
import time
import math

#allowing V1 behavior while using Tf 2.0
tf.disable_v2_behavior()

#reading the arguments
arglist = sys.argv

path = arglist[1]
falsepositive = arglist[7]
small_thresh = arglist[2]
large_thresh = arglist[3]

large_filter = int(arglist[4])
medium_filter = int(arglist[5])
small_filter = int(arglist[6])

#creating the region file and the catalog file
regionfile = open('results/'+path[6:len(path)-15]+'.reg','w+')
catalog = open('results/'+path[6:len(path)-15]+'_catalog.txt','w+')

#applies guassian Blur and thresholds the image
def filter(img): 
	blur = cv2.GaussianBlur(img, (5,5),0)
	retval, threshold = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
	return(threshold)

#load the MobileNet's graph
def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()
	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)
	return(graph)

#creates image into a tensor for the MobileNets
def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255): 
	input_name = 'file_reader'
	output_name = 'normalized'
	file_reader = tf.read_file(file_name, input_name)
	image_reader = tf.image.decode_png(file_reader, channels = 3, name = 'png_reader')
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0);
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	result = sess.run(normalized)
	return(result)

#converts the Lat coordinate into format of "000.00"
def convertLat(lat): 
	lat = str(lat)
	pos = lat.find('.')
	a = lat[pos+1]
	b = lat[pos+2]
	x = '0'
	y = '0'
	z = '0'
	if(pos-1>=0):
		z = lat[pos-1]
	if(pos-2>=0):
		y = lat[pos-2]
	if(pos-3>=0):
		x = lat[pos-3]
	return(x+y+z+'.'+a+b)

#converts the Lon coordinate into format of "(+,-)00.00"
def convertLon(lon_og): 
	lon = str(lon_og)
	pos = lon.find('.')
	a = lon[pos+1]
	b = lon[pos+2]
	y = '0'
	z = '0'
	if(pos-1>=0):
		z = lon[pos-1]
	if(pos-2>=0):
		y = lon[pos-2]
		if(y == '-'):
			y = '0'
	if(lon_og<0):
		return('-'+y+z+'.'+a+b)
	else:
		return('+'+y+z+'.'+a+b)

#creating tf session
sess = tf.Session() 

#define small MobileNet 
model_file = "assets/retrained_graph.pb"
label_file = "assets/retrained_labels.txt"
input_height = 128
input_width = 128
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"

#define large MobileNet
model_file2 = "assets/retrained_graph2.pb"
label_file2 = "assets/retrained_labels2.txt"
input_height2 = 224
input_width2 = 224
input_mean2 = 224
input_std2 = 224
input_layer2 = "input"
output_layer2 = "final_result"

graph = load_graph(model_file)

input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);

graph2 = load_graph(model_file2)

input_name2 = "import/" + input_layer2
output_name2 = "import/" + output_layer2
input_operation2 = graph2.get_operation_by_name(input_name2);
output_operation2 = graph2.get_operation_by_name(output_name2);

#Tf sessions for the MobileNets
sess2 = tf.Session(graph=graph)
sess3 = tf.Session(graph=graph2)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
label_lines = [line.rstrip() for line  in tf.gfile.GFile(label_file)]

with tf.gfile.FastGFile(model_file, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')

with tf.gfile.FastGFile(model_file2, 'rb') as f2:
	graph_def2 = tf.GraphDef()
	graph_def2.ParseFromString(f2.read())
	tf.import_graph_def(graph_def2, name='')

#getting image data 
print(path, "is being processed")
im = fits.open(path)
#sizex
sx = im[0].header['NAXIS1'] 
#sizey
sy = im[0].header['NAXIS2'] 
header = fits.getheader(path)
#coordinate data
coor = WCS(header) 
image_data = im[0].data
#convert to image
img = Image.fromarray(image_data) 
image_data = fits.getdata(path)

img = img.convert('RGB')
height_limit = sy
#save image
img.save('images/img.png') 
#open image with opencv
img = cv2.imread('images/img.png',0) 
#applying contrast limited adaptive histogram equalization
clahe = cv2.createCLAHE(tileGridSize=(8,8)) 
equ = clahe.apply(img)
#save the new image
cv2.imwrite('images/img.png',equ) 

#opening the new image with both OpenCV and PIL
im_path = 'images/img.png'
pil_im = Image.open(im_path)
img = cv2.imread(im_path)
#apply the filters to the image
img = filter(img)
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),10, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

boxes = []
#obtain all the contours
for c in range(len(contours)):

	x, y, w, h = cv2.boundingRect(contours[c])
	areaa = cv2.contourArea(contours[c])   
	boxes.append([x, y, w, h,w*h,contours[c]])

#sort and keep only the 1000 largest contours
boxes = sorted(boxes, key=itemgetter(4))
boxes.remove(boxes[len(boxes)-1])
boxes = boxes[::-1]
boxes = boxes[:1000]
newbox = [] 

for i in boxes:

	area = cv2.contourArea(i[5])

	#calculate the color difference
	smallbox = pil_im.crop((i[0]+i[2]/3,i[1]+i[3]/3,i[0]+i[2]-i[2]/3,i[1]+i[3]-i[3]/3))
	largebox = pil_im.crop((i[0]-15,i[1]-15,i[0]+i[2]+15,i[1]+i[3]+15))
	avgrow = np.average(largebox, axis=0)
	avgc = np.average(avgrow, axis=0)
	avg_color_per_row = np.average(smallbox, axis=0)
	avg_color = np.average(avg_color_per_row, axis=0)
	difference = avgc-avg_color

	#calculate the depth
	regularbox = image_data[i[1]:i[1]+i[3], i[0]:i[0]+i[2]]
	B1 = image_data[i[1]-10:i[1]+i[3]+10, i[0]-10:i[0]].flatten()
	B2 = image_data[i[1]-10:i[1], i[0]+1:i[0]+i[2]-1].flatten()
	B3 = image_data[i[1]-10:i[1]+i[3]+10, i[0]+i[2]:i[0]+i[2]+10].flatten()
	B4 = image_data[i[1]+i[3]:i[1]+i[3]+10, i[0]+1:i[0]+i[2]-1].flatten()
	values = np.concatenate((B1, B2, B3, B4), axis=None)
	values = values[~np.isnan(values)]
	regularbox = regularbox[~np.isnan(regularbox)]
	minimum = np.min(regularbox)
	median = np.median(values)
	i.append(median-minimum)
	i.append(area)

	if(falsepositive == '1'):
		if(((i[1] >= .3*height_limit) and (i[1] <= .7*height_limit)) or area > 1000):
			if( i[4] < 500000):
				if(area > 30000 and difference > large_filter):
					newbox.append(i)
				if(area <= 30000 and area >= 3000 and difference > medium_filter):
					newbox.append(i)
				if(area < 3000 and difference > small_filter):
					newbox.append(i)
	else:
		if( i[4] < 500000):
			if(area > 30000 and difference > large_filter):
				newbox.append(i)
			if(area <= 30000 and area >= 3000 and difference > medium_filter):
				newbox.append(i)
			if(area < 3000 and difference > small_filter):
				newbox.append(i)


#number of small and large IRDCs (small = bounding box area less than 30000)
num_small_irdc = 0
num_large_irdc = 0

for c in newbox:

	#calculating the center coordinates and the width and height in both arcsec and degrees
	midx = round(c[0]+float(c[2])/2)
	midy = round(c[1]+float(c[3])/2)
	midx = midx
	midy = midy
	fx = round(sx*(midx/float(sx)))
	fy = round(sy*(midy/sy))
	lat, lon = coor.all_pix2world(fx,fy,1)
	startx = sx*(c[0]/float(sx))
	starty = sy*(c[1]/float(sy))
	endx = round(sx*((c[0]+c[2])/float(sx)))
	endy = round(sy*((c[1]+c[3])/float(sy)))
	lat1, lon1 = coor.all_pix2world(startx,starty,1)
	lat2, lon2 = coor.all_pix2world(endx,endy,1)
	dlat = abs(lat1-lat2)
	dlon = abs(lon1-lon2)
	width = c[2]
	height = c[3]
	width = width*1.2
	height = height*1.2
	width_degrees = width*0.00027777777
	height_degrees = height*0.00027777777

	# Small MobileNet
	if(c[2]*c[3] < 30000):  
		#obtain the cropped image of the bounding box and run it through the MobileNet
		cropped = pil_im.crop((c[0]-5, c[1]-5,c[0]+c[2]+5,c[1]+c[3]+5))
		cropped.save('images/im.jpeg')
		cropped = cv2.imread('images/im.jpeg',0)
		tensor = read_tensor_from_image_file('images/im.jpeg',
		input_height=input_height, 
		input_width=input_width,
		input_mean=input_mean,
		input_std=input_std)
		results = sess2.run(output_operation.outputs[0],{input_operation.outputs[0]: tensor})
		results = np.squeeze(results)

		if(results[1]>float(small_thresh)):
			print('G'+convertLat(lat)+convertLon(lon)+' '+str(format(lat, '.4f'))+' '+str(format(lon, '.4f'))+' '+str(format(width, '.1f'))+' '+str(format(height, '.1f'))+' '+str(format(c[6], '.2f'))+' '+str(format(c[7], '.0f')))
			num_small_irdc += 1
			regionfile.write('Galactic;'+' box '+str(lat)+' '+str(lon)+' '+str(width_degrees)+' '+str(height_degrees)+'\n')
			catalog.write('G'+convertLat(lat)+convertLon(lon)+' '+str(format(lat, '.4f'))+' '+str(format(lon, '.4f'))+' '+str(format(width, '.1f'))+' '+str(format(height, '.1f'))+' '+str(format(c[6], '.2f'))+' '+str(format(c[7], '.0f'))+'\n')
	
	#Large MobilNet
	else:
		#obtain the cropped image of the bounding box and run it through the MobileNet
		cropped = pil_im.crop((c[0]-5, c[1]-5,c[0]+c[2]+5,c[1]+c[3]+5))
		cropped.save('images/im.jpeg')
		cropped = cv2.imread('images/im.jpeg',0)
		tensor = read_tensor_from_image_file('images/im.jpeg',
		input_height=input_height, 
		input_width=input_width,
		input_mean=input_mean,
		input_std=input_std)
		results = sess2.run(output_operation.outputs[0],{input_operation.outputs[0]: tensor})
		results = np.squeeze(results)

		if(results[1]>float(large_thresh)):
			print('G'+convertLat(lat)+convertLon(lon)+' '+str(format(lat, '.4f'))+' '+str(format(lon, '.4f'))+' '+str(format(width, '.1f'))+' '+str(format(height, '.1f'))+' '+str(format(c[6], '.2f'))+' '+str(format(c[7], '.0f')))
			num_large_irdc += 1
			regionfile.write('Galactic;'+' box '+str(lat)+' '+str(lon)+' '+str(width_degrees)+' '+str(height_degrees)+'\n')
			catalog.write('G'+convertLat(lat)+convertLon(lon)+' '+str(format(lat, '.4f'))+' '+str(format(lon, '.4f'))+' '+str(format(width, '.1f'))+' '+str(format(height, '.1f'))+' '+str(format(c[6], '.2f'))+' '+str(format(c[7], '.0f'))+'\n')

print(num_small_irdc," small IRDCs found  -  ",num_large_irdc," large IRDCs found  -  ",num_large_irdc+num_small_irdc, " found in total")



