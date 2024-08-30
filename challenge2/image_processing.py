import cv2
from cv2 import dnn_superres

# initialize super resolution object
sr = dnn_superres.DnnSuperResImpl_create()

# read the model
path = r'C:\Users\arifz\Downloads\sunway\challenge2\EDSR_x4.pb'
sr.readModel(path)

# set the model and scale
sr.setModel('edsr', 4)

# load the image
image = cv2.imread(r'C:\Users\arifz\Downloads\sunway\challenge2\youngsters.jpg')

# upsample the image
upscaled = sr.upsample(image)
# save the upscaled image
cv2.imwrite('enhance_youngster_EDSR.png', upscaled)