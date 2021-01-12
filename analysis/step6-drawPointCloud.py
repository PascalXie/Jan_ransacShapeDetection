import math
import numpy as np
import matplotlib.pyplot as plt

def ReadFile(path, fileName):
	print("Data File Path : {}".format(path))
	print("File Name : {}".format(fileName))

	# read
	f = open(path+fileName)
	lines = f.readlines()

	X		= []
	Y		= []
	Z		= []
	NorX	= []
	NorY	= []
	NorZ	= []

	counter = 0
	for line in lines:
		line = line.strip().split()

		x		= float(line[0])
		y 	  	= float(line[1])
		z	  	= float(line[2])
		norx  	= float(line[3])
		nory  	= float(line[4])
		norz  	= float(line[5])

		X.append(x)
		Y.append(y)
		Z.append(z)
		NorX.append(norx)
		NorY.append(nory)
		NorZ.append(norz)
	
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	NorX = np.array(NorX)
	NorY = np.array(NorY)
	NorZ = np.array(NorZ)

	return X, Y, Z, NorX, NorY, NorZ

if __name__=='__main__':
	print('hello')

	path = '../step6-shapeDetect/step1-generateCylinders/build-cylinderGenerator-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'cylinders.txt'
	X, Y, Z, NorX, NorY, NorZ = ReadFile(path, fileName)

	# figure
	fig = plt.figure(dpi=128,figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')

	# draw
	ax.scatter(X, Y, Z, s=1, cmap="jet", marker="o")

	# set lable 
	ax.set_xlabel('X', fontsize=10)
	ax.set_ylabel('Y', fontsize=10)
	ax.set_zlabel('Z', fontsize=10)

	# set limits
	#ax.set_xlim(-200, 200)
	#ax.set_ylim(-200,200)
	#ax.set_zlim(-150, 50)

	# set title
	plt.title('Original Point Cloud', fontsize=10)
	
	# overlapping 
	plt.tight_layout()

	# legend
	plt.legend()

	# save figure
	plt.savefig('figure_step6_pointCloud.png')

	# print figure on screen
	plt.show()

