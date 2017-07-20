# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Detect and highlight both side of lines of lane in red color
* Process images in the testing image folder, and save the results in the output folder.
* Process each frame in the video using the pipeline, and save the processed results in the video output folder


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps as follows:
 * Converting the input image to grayscale
 * Bluring the image with Gaussian kernels
 * Finding the edges with Canny filter
 * Finding the lines with Hough Transform from the edges image
 * Selecting the lines with orientation selectivity 
 * Clipping the region of interest (ROI) using a polygon
 * Applying the Hough Transform the second time to further enhance the line connectivity
 * Merging lines to a single line for left and right lanes

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
