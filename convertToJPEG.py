import cv2
import numpy as np
from scipy import misc

for i in range(21, 41):
	image=misc.imread("training/1st_manual/"+str(i)+"_training_mask.gif")
	misc.imsave("training/mask/"+ str(i-20) + ".JPEG", image)
