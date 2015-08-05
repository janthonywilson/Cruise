Cruise Challenge #1
===================

Introduction
------------

Thanks for agreeing to spend some time hacking on computer vision with us.  This challenge is designed to be fun, realistic, and possibly quite difficult.  You will write code to visually identify lane markers in sequences highway images.  We will evaluate your solution bases on accuracy, code quality, and how you clearly you describe your technique.

Happy hacking!

Data Set
--------

We've supplied a sequence of images for you to test against.  They represent highway lane markers seen in real world conditions, which means your algorithm may work well in some cases and poorly in others.  Robust code is a necessity when dealing with unstructured environments, and that kind of code is hard to write.  Your goal should be to correctly identify the lane markers in as many images as possible, but we don't expect you to get them all exactly right.

The data set is broken into sequences of images.  The images are chronologically ordered and cover continuous sections of highway, so feel free to use this to your advantage.

We have supplied a code stub in lane_track.py, which can be used as an example to get you up and running quickly.

The Challenge
-------------

Modify the code stub called 'track.py' to process the images in the sample data set and produce a CSV file called 'intercepts.csv' containing the estimated x-intercept of the left and right lane markers closest to the car. 

For example, if you process the file 'mono_000000028.png' and calcuate that the left lane marker would hit the x-intercept (bottom of the image) at the 102nd pixel from the left and the right marker at the 708th pixel from the left, you'd write the following line in your CSV file:

mono_000000028.png,102,708

Your CSV file should contain one line for each image in the images folder.  If you can't determine the location of one or both of the lane markers, just write the value 'None' instead of the x-intercept.

When run, your code should also display each image and overlay the detected lane edge and/or lane markings in yellow. The yellow markings can be displayed as line segments, splines, or however you see fit.

Submission
----------

Please submit the following:

  1. Your code as one or more python or C++ files
  2. Your intercepts.csv file
  3. A brief writeup of how your code works
  4. What you'd do next to improve it if you had more time

