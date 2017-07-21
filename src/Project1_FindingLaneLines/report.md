# **Project 1: Finding Lane Lines on the Road** 

---

**Goal and Steps**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Detect and highlight both side of lines of lane in red color
* Process images in the testing image folder, and save the results in the output folder.
* Process each frame in the video using the pipeline, and save the processed results in the video output folder


[//]: # (Image References)

[input]: ./test_images/whiteCarLaneSwitch.jpg "Input"
[gray]: ./test_images_output/whiteCarLaneSwitch-gray.jpg "Gray"
[blur_gray]: ./test_images_output/whiteCarLaneSwitch-blur_gray.jpg "Blur Gray"
[color_edge]: ./test_images_output/whiteCarLaneSwitch-color_edge.jpg "Color Edge"
[edges]: ./test_images_output/whiteCarLaneSwitch-edges.jpg "Edges"
[final]: ./test_images_output/whiteCarLaneSwitch-final.jpg "Final"
[lines0]: ./test_images_output/whiteCarLaneSwitch-lines0.jpg "Lines"
[lines]: ./test_images_output/whiteCarLaneSwitch-lines.jpg "Lines"
[roi1]: ./test_images_output/whiteCarLaneSwitch-roi1.jpg "ROI 1"
[roi2]: ./test_images_output/whiteCarLaneSwitch-roi2.jpg "ROI 2"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of seven steps as follows:
 * Step 1: Converting the input image to grayscale
 ![alt text][gray]
 * Step 2: Bluring the image with Gaussian kernels
 ![alt text][blur_gray]
 * Step 3: Finding the edges with Canny filter
 ![alt text][edges]
 * Step 4: Finding the lines with Hough Transform from the edges image
 ![alt text][lines0]
 * Step 5: Selecting the lines with orientation selectivity
 ![alt text][lines]
 * Step 6: Clipping the region of interest (ROI) using a polygon
 ![alt text][roi1]
 * Step 7: Applying the Hough Transform the second time to further enhance the line connectivity, 
 and merging lines to a single line for left and right lanes
 ![alt text][roi2]
 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following operations:
* Line selection by their orientation. See function line_selection
* Merging and draw a single line by MergeLines function, including the steps below:
    * Clustering lines to left and right clusters (see function LineCluster)
    * Extending lines to the polygon boundary (see function LinearExtension)
    * Validing lines by checking the ratio of line falling into the left portion (see function LeftRatio)

Here is the final output:
 ![alt text][final]

### 2. Identify potential shortcomings with your current pipeline

The main limitations of current implementation is depending on many hard-coded parameters. In detail:

* Hardcoded polygon, the ROI.
* The lane are not accurately detected in certain frames, such as in the "Challenge" video, the frames 110-119.
* Lane is shaking across frames, needs a stabilizer
* The clustering of left- or right-side of lane lines are done with hard-coded parameter. This can be smarter. 
* The grouping of the lines to left or right based on the orientation, but the orientation is hard-coded.

### 3. Suggest possible improvements to your pipeline

Here are a few items that I can think of to improve at current point:

* Automatic ROI detection, including:
    * Horizon line detection
    * Top x1, x2 detection
* Image normalization (maybe using HSV space?)
* Line stabilizer between frames
* Better validation of recognized lines. For example, check the width changes across frames.
