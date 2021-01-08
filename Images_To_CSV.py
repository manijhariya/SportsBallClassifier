import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

def load_image(location,flatten=True):
	img = None
	try:
		img = Image.open(location).convert('RGB')
		img  = img.resize((100,100))
		img = np.asarray(img,dtype=np.int)
		#plt.imshow(img ,interpolation='nearest')
		#plt.show()
		img = img / 255.0
		if flatten:
			img = img.flatten()
		#print (img.shape)
	except Exception as err:
		print (str(err))
		img = None

	return img

def file_list():
	a = os.listdir('Data/')
	return a

def main():
	a = file_list()
	Imagedata = []
	for filename in a:
		if not (filename.endswith('.jpeg') or filename.endswith('.png')):
			continue

		filename = ('Data/%s'%filename)
		data = load_image(filename)
		if data is not None:
			if "golf" in filename:
				data = np.append(data,0.0)
			elif "basketball" in filename:
				data = np.append(data,1.0)
			elif "soccer" in filename:
				data = np.append(data,2.0)
			else:
				continue
		else:
			continue
		Imagedata.append(data)


	df = pd.DataFrame(Imagedata)
	df.to_csv("ImageCSV.csv",index=False)
	print (df.shape)
	print ("Done")

if __name__ == "__main__":
	main()
