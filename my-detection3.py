import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)

image_paths = [
    "/home/nvidia/Desktop/image1.jpg",
    "/home/nvidia/Desktop/image2.jpg"
]

for img_path in image_paths:

    img = jetson.utils.loadImage(img_path)

    detections = net.Detect(img)

    for i, detection in enumerate(detections):

        print(detections[i])

