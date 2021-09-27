
# Notes on Multiple Object Tracking

Multiple Object Tracking = MOT

### Tracking-by-detection
*Tracking-by-detection* is the standard MOT approach, which consists of detection and association.  In detection, the goal is to come up with bounding boxes around objects in each frame.  Association is the process by which bounding boxes from different frames are assigned to each other, hence tracking objects as they move.  If you would like to compared different methods of association, then you should keep the object detection method constant, and vice versa.



