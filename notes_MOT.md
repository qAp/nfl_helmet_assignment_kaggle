
# Notes on Multiple Object Tracking

Multiple Object Tracking = MOT

### Tracking-by-detection
*Tracking-by-detection* is the standard MOT approach, which consists of *detection* and *association*.  In detection, the goal is to come up with bounding boxes around objects in each frame.  Association is the process by which bounding boxes from different frames are assigned to each other, hence tracking objects as they move.  

If you wanted to compare different methods for the association step, then you would keep the object detection model used constant, and vice versa.  Hence, some MOT datasets come with bounding boxes predicted by some object detection model, which means that the object detection step can be skipped, and focus can be put on improving the association step.

### Batch and online methods
MOT algorithms can also be divided into *batch* and *online* methods.  To make predictions for the present, batch methods make use of both information from the past and the future, while online methods only use information from the past and present.  Because of this difference, batch methods are more accurate.  However, online methods are needed for scenarios such as autonomous driving, and robotics. 




