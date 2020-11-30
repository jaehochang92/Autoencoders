import cv2, os, time
import numpy as np
from pprint import pprint
from tqdm import tqdm
# from pytictoc import TicToc
from imgaug import augmenters as ia
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def draw_bounding_box(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# function to get the output layer names 
# in the architecture
def get_output_layers(net): 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# open the video file
def extract_frames(cap, lim):
    if not os.path.exists('frames'):
        print('Created folder "frames" on current working directory')
        os.mkdir('frames')
    ret, frame = cap.read()
    index = 0
    while ret and index < lim:
        cv2.imwrite(f"frames/frame{index:07d}.jpg", frame)
        ret, frame = cap.read()
        index += 1

def aug_img(img, augmenter):
    # ia.imshow(img)
    auged_img = augmenter(image = img)
    return auged_img

def prprdata(w, h, fs, IMAGES, augmenters_c):
    index = 1
    frames = []
    aimgs = []

    resizer = ia.size.Resize({"height": h, "width": w})
    gray_scaler = ia.Grayscale()
    print(resizer)
    
    for IMAGE in IMAGES:
        for _, augmenters in augmenters_c.items():
            for augmenter_name, augmenter in augmenters.items():
                cap = cv2.VideoCapture(IMAGE)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                noisy_out = cv2.VideoWriter(f"{augmenter_name}_{IMAGE.split('/')[-1].split('.')[0]}.avi", fourcc, 5, (w, h))
                clean_out = cv2.VideoWriter(f"{IMAGE.split('/')[-1].split('.')[0]}.avi", fourcc, 5, (w, h))
                ret, frame = cap.read()
                k = int(8000 / fs)
                while ret and index < fs * k:
                    if index % fs == 0:
                        frame = resizer.augment_image(frame)
#                         frame = gray_scaler.augment_image(frame)
                        frames.append(frame)
                        clean_out.write(frame)

                        aimg = aug_img(frame, augmenter)
                        aimgs.append(aimg)
                        noisy_out.write(aimg)
                    index += 1
                    ret, frame = cap.read()
                index = 1
                
    cap.release()
    clean_out.release()
    noisy_out.release()
    cv2.destroyAllWindows()
    
    return frames, aimgs