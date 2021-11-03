# Notes on the FairMOT paper

## 2-model MOT methods
Many existing methods for MOT consist of 2 separate models: the detection model and the association model.  

The detection model locates objects of interests in each video frame, by placing a bounding box around each of them.  e.g. Mask R-CNN, SSD, and Yolo, etc. are object detection models that can do this.

The association model does two things.  It first extracts re-identification (re-ID) features from each bounding box, and then uses these features to link each object to existing tracks, according to certain metrics defined on these features.  

## 1-model MOT methods
With the advent of multi-task learning, there are methods that combine the detection and association tasks together, such that a single network learns both the features for object detection and association.  There are called one-shot methods.  e.g. Mask R-CNN, an object detection model, can have a branch added to it that is responsible for learning re-ID features using ROI-Align.  However, even though one-shot methods, by using a single network, are much faster during inference, their tracking performance have lagged behind the best 2-model MOT methods.  

In this work on FairMOT, they investigate the reasons behind this and propose an improved solution. 
