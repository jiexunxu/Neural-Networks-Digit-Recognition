# Introduction

This is a Java implementation of the convolutional network from scratch. The project is used for digit 0~9 recognition. It can achieve 93.4% overall accuracy.

# Structure

The Java project contains ten files. 
- C_layer and S_layer are the convolution and subsampling layers.
- Classifier is the classifier that classifies the input.
- Network class assembles C_layer, S_layer and Classifier.
- Vector, Matrix and image classes are supporting classes to simplify implementation.
- Initializer initializes a parameter file to record kernels, bias and scalar multipliers.
- InfoReader reads information from the parameter file.

# Setup

To compile the program, type “javac *.java” at the command line (you need JDK1.6)

To run the program, type “java Test_Network filename”, where filename is one of the input test files provided in the zipped file and this file is in the same folder as the project.

Input test files are text files composed of pixel values of a 36*36 image. The first digit in the filename represents the digit this file is representing(ex: b_0_0, b_0_1 etc means this file represents digit 0). So to test b_0_0.txt, type “java Test_Network b_0_0.txt”.

The dataset contains 6000 training samples and 1000 test samples for each digit. The network has three convolution layers and three subsampling layers.First convolution layer has 6 output images, second layer has 16 images and the third has 120 images.

# Result
The convolutional neural network was trained for about 100,000 samples in total, and then tested agaisnt 1000 samples for each digit. 

Following table is the result:

| Digit | Accuracy |
| ----- | -------- |
| 0 | 97.8% |
| 1 | 98.5% |
| 2 | 95.2% |
| 3 | 90.3% |
| 4 | 95.5% |
| 5 | 97.4% |
| 6 | 92.8% |
| 7 | 91.9% |
| 8 | 88.8% |
| 9 | 85.9% |
| Overall | 93.4% |