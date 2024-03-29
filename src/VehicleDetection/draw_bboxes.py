import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes,
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for P1, P2 in bboxes:
        x1, y1 = P1
        x2, y2 = P2
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((279, 497), (372, 563)),
            ((483, 508), (543, 553)),
            ((593, 511), (633, 543)),
            ((645, 511), (677, 534)),
            ((839, 501), (1114, 673))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
