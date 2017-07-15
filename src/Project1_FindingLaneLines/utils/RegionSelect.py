import numpy as np
import cv2

def ColorSelect(image, rgb_threshold):
    color_select = np.copy(image)
    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    color_mask = (image[:, :, 0] < rgb_threshold[0]) \
                 | (image[:, :, 1] < rgb_threshold[1]) \
                 | (image[:, :, 2] < rgb_threshold[2])
    color_select[color_mask] = [0, 0, 0]
    return color_select, color_mask


# Define a triangle region of interest
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
def TriangularRegionSelect(image, left_bottom, right_bottom, apex):
    region_select = np.copy(image)
    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]

    # Fit lines (y=Ax+B) to identify the  3 sided region of interest
    # np.polyfit() returns the coefficients [A, B] of the fit
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_mask = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    # Color pixels red which are inside the region of interest
    region_select[region_mask] = [255, 0, 0]
    return region_select, region_mask

def PolyRegionSelect(image, vertices):
    region_mask = np.zeros(image.shape)
    region_select = cv2.fillPoly(image, vertices, [255, 0, 0])
    region_mask = cv2.fillPoly(region_mask, vertices, [1, 0, 0])[:,:,0].astype(bool)
    return region_select, region_mask


def TriangularRegionAndColorSelect(image, left_bottom, right_bottom, apex, rgb_threshold):
    line_image = np.copy(image)

    region_select, region_mask = TriangularRegionSelect(image, left_bottom, right_bottom, apex)

    # Mask pixels below the threshold
    color_select, color_mask = ColorSelect(image, rgb_threshold)

    # Mask color and region selection
    color_select[color_mask | ~region_mask] = [0, 0, 0]
    # Color pixels red where both color and region selections met
    line_image[~color_mask & region_mask] = [255, 0, 0]

    return line_image, color_select
