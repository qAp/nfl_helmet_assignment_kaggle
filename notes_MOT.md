
# Notes on Multiple Object Tracking

Multiple Object Tracking = MOT

## Tracking-by-detection
*Tracking-by-detection* is the standard MOT approach, which consists of *detection* and *association*.  In detection, the goal is to come up with bounding boxes around objects in each frame.  Association is the process by which bounding boxes from different frames are assigned to each other, hence tracking objects as they move.  

If you wanted to compare different methods for the association step, then you would keep the object detection model used constant, and vice versa.  Hence, some MOT datasets come with bounding boxes predicted by some object detection model, which means that the object detection step can be skipped, and focus can be put on improving the association step.

## Batch and online methods
MOT algorithms can also be divided into *batch* and *online* methods.  To make predictions for the present, batch methods make use of both information from the past and the future, while online methods only use information from the past and present.  Because of this difference, batch methods are more accurate.  However, online methods are needed for scenarios such as autonomous driving, and robotics. 

## Main stages of an MOT algorithm
1. Detection stage
2. Feature extraction/motion prediction stage
3. Affinity stage
4. Association stage


## Visual feature extraction using CNNs

CNN extracts visual features.
Kalman filter extracts position and/or motion features.
Different kinds of features can be combined before being passed onto the affinity stage.  


Fu et al. [98]
1. DeepSORT feature extracter extracts features.
2. Discriminative correlation filter measures the correlation between the features.  This gives a matching score.
3. The matching score from 2 is combined with spatio-temporal relation score.
4. Combined score from 3 is used as a likelihood in a Guassian Mixture Probability Hypothesis Density filter.
