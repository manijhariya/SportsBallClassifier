from tensorflow.keras.models import load_model
from Images_To_CSV import load_image
import numpy as np

class PREDICT_MODEL:
	def __init__(self,name):
		if name == "CNN":
			self.model = load_model("CNN_model.h5")

	def predict(self,location):
		img = load_image(location,flatten=False)
		img = img.reshape((1,100,100,3))
		result = self.model.predict(img)
		result = np.argmax(result[0])
		if result == 0:
			return ("Golf Ball")
		elif result == 1:
			return ("Basket Ball")
		elif result == 2:
			return ("Soccer Ball")
		else:
			return ("Sorry Can't determine")
