import dxcam

camera = dxcam.create()
frame = camera.grab()

from PIL import Image
Image.fromarray(frame).show()
