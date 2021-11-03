# A solution for NFL Health & Safety - Helmet Assignment Kaggle

https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/overview

## Solution Overview
1. Multiple Object Tracking: Train FairMOT to track helmets in videos.
<video width="320" height="240" controls>
  <source src="https://www.kaggleusercontent.com/kf/78474745/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..lXMmNK_lgit9z2fzX5J35g.URQeorKPkuNwwOmewWimH09wJlD1LpxDVtp7-BlstKkf1gbZvnsYUeeWUUxJdMHbO7BB4zwWdhPs8bSlMWy1IwPEHmHH1D-XRTn9BBtAXFiFd9vXFCMogBUlXBUjK6DAI_0Rs7bdotHfpyHXaVnKFjEZN8MuQwL8cEm9SR1UMQFoluiYOEeKyJBPnLY0YcxKGtrje1xIaUcoGzDq4RGOc5HOUHgPWpldCgG0EEPNBM4Dr_LkUW5e18rHoreISPuQ_Vb3j_Ya3vbRe1PXOlNDDS5Xx4FBy_qY0hwf7xf8wqdoO-MjCGbz4i3Rb4RXxt91zis9dkE00WkRKK39ftkclP_YNf6Jz3xhrqrODzOAnML91WSvZRe4h6GYnPoyzsCv1nrPWiAHWyjRswWnENJm0N3HqEQLTMVOZiVqraRqGxiS2l4Xutp0uUm1wBvuLZI4QJUizdGRVdrqT-MFMV3jz-fLdDMCfsaY7pf4AHjocNXmgCnpNx00-NR4y5YCu2EAXw3D0FVX4UX4EedQJDxy_kykg4tneikG_aYpUG1XA49tBJ6eg77_u4COtVnO8keXQUYpD04e0y8jXbZhJqzYq3aB3scicvNl9KJuzm2JRoRpvY3jztc69w3Cc7veHB_K8cu--vth-Pp6-0YgFnfKxRv4y9Nl_PqA6NK43NIbmSU2AOKmMawzs18mio8IzJqd.BwL-P7jKwN_7dFm9zIkUcg/demo_debug/57700_001264_Endzone/results.mp4" type="video/mp4">
</video>

2. Helmet mapping: Map NGS player labels to baseline helmet detections.
3. Incorporate FairMOT tracks into helmet mapping labels.
4. Fill in missing labels.

## TODOs
- [x] Tune helmet mapping
- [x] Use y-coordinate as well in helmet mapping.
- [x] Tune FairMOT
- [x] Evaluate helmet mapping (1D and 2D) + FairMOT (latest)
- [x] Construct inference pipeline: helmet mapping followed by FairMOT
- [x] Make inference notebook work for submission.
- [x] Don't ffmpeg convert demo's output videos to friendlier formats for submission.
- [x] Tweak FairMOT post-processing: (duplicated MOT ids, etc.)
- [ ] Post-process with DeepSORT then FairMOT
- [ ] Slow down the video at helmet impacts for FairMOT training and inference

## Filtering out excess NGS positions
When the angle of rotation is good, the filtering algorithm appears to filter out the correct players that are out of view in the camera:
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/85662dac99363341185d1edf4014c86fe0ac791b/images/ngs_players_filtering/good_theta_good_filtering.png" width="800">

When the angle of rotation isn't good, the incorrect players may be left out:
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/85662dac99363341185d1edf4014c86fe0ac791b/images/ngs_players_filtering/bad_theta_bad_filtering.png" width="800">

## Is the camera on the home or away side?
The competition data tells us whether the camera is on a sideline or an endzone, but it doesn't tell us if it's the home or away sideline/endzone.  So, when mapping helmets to player numbers, it is necessary to assume that the camera can be on the home or away side, and then to perturb the camera angle from those assumed initial orientations to get the orientation that produces the best match between the helmet positions and the NGS player positions.

Suppose we know that the camera is on an endzone, then we need to try both the home and away endzone.  However, only one of these is where the camera actually is, so you would expect that the computed distance score to be smaller for the true side.  

Below are the minimum distance scores for the 4 possible true pitch sides (e.g. home sideline: 0 degrees; home endzone: 90 degrees; etc.), in selected video frames.  The corresponding perturbation angle is shown in the right panel.  The blue lines are for the true side, where the NGS reference frame is more or less aligned with a camera positioned on this side.  The orange lines are for the side opposite the true side.  

True pitch side for camera: home sideline, NGS rotated by 0 deg.
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/5290172436e765c322372f86aed78733b10a8378/images/home_or_away/home_away_tinit0.png" width="600">

True pitch side for camera: home endzone, NGS rotated by 90 deg.
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/5290172436e765c322372f86aed78733b10a8378/images/home_or_away/home_away_tinit90.png" width="600">

True pitch side for camera: away sideline, NGS rotated by 180 deg.
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/5290172436e765c322372f86aed78733b10a8378/images/home_or_away/home_away_tinit180.png" width="600">

True pitch side for camera: away endzone, NGS rotated by 270 deg.
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/5290172436e765c322372f86aed78733b10a8378/images/home_or_away/home_away_tinit270.png" width="600">

The blue lines of the distance score appear to be mostly below the orange lines, making it obvious to determine which the correct side is, but there are some overlaps, perhaps when the players are in a near symmetrical configuration.

## Fine-tuning FairMOT



## Evaluation on selected train samples

|    |  hmap | hmap + pretrained DeepSORT | hmap + pretrained FairMOT | hmap + finetuned FairMOT |
|:--:|:-----:|:--------------------------:|:-------------------------:|:------------------------:|
| 1d | 0.288 |            0.446           |           0.438           |           0.513          |
| 2d | 0.312 |            0.459           |                           |           0.520          |

Here, due to randomness introduced in helmet mapping, all scores based on MOT post-processing use the same instance of helmet mapping result.  e.g. all 1-d post-processing scores are built upon the 0.288 helmet mapping score.  


## References
- https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
- https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping
- https://www.kaggle.com/robikscube/helper-code-helmet-mapping-deepsort
- SORT paper: https://arxiv.org/pdf/1602.00763.pdf
- DeepSORT paper: https://arxiv.org/pdf/1703.07402.pdf
- Deep Learning in Video Multi-Object Tracking: A Survey: https://arxiv.org/abs/1907.12740
- FairMOT paper: https://arxiv.org/pdf/2004.01888.pdf
- FairMOT on GitHub: https://github.com/ifzhang/FairMOT




