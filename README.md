# Code for NFL Health & Safety - Helmet Assignment Kaggle

## TODOs
- [] Tune helmet tracking
- [] Tune FairMOT


## Filtering out excess NGS positions
When the angle of rotation is good, the filtering algorithm appears to filter out the correct players that are out of view in the camera:
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/85662dac99363341185d1edf4014c86fe0ac791b/images/ngs_players_filtering/good_theta_good_filtering.png" width="800">

When the angle of rotation isn't good, the incorrect players may be left out:
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/85662dac99363341185d1edf4014c86fe0ac791b/images/ngs_players_filtering/bad_theta_bad_filtering.png" width="800">

## Is the camera on the home or away side?
The competition data tells us whether the camera is on a sideline or an endzone, but it doesn't tell us if it's the home or away sideline/endzone.  So, when mapping helmets to player numbers, it is necessary to assume that the camera can be on the home or away side, and then to perturb the camera angle from those assumed initial orientations to get the orientation that produces the best match between the helmet positions and the NGS player positions.

Suppose we know that the camera is on an endzone, then we need to try both the home and away endzone.  However, only one of these is where the camera actually is, so you would expect that the computed distance score to be smaller for the true side.  

Unfortunately, this does not appear to be obvious.  Below are the minimum distance scores for the 4 possible true pitch sides (e.g. home sideline: 0 degrees; home endzone: 90 degrees; etc.), in selected video frames.  The corresponding perturbation angle is shown in the right panel.  The blue lines are for the true side, where the NGS reference frame is more or less aligned with a camera positioned on this side.  The orange lines are for the side opposite the true side.  

<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/84e3820869f84d85caa04b4a32ff45a711ffdd02/images/checking_this_or_that_side/this_that_true0.png" width="800">

## References
- https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
- https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping
- https://www.kaggle.com/robikscube/helper-code-helmet-mapping-deepsort
- SORT paper: https://arxiv.org/pdf/1602.00763.pdf
- DeepSORT paper: https://arxiv.org/pdf/1703.07402.pdf
- Deep Learning in Video Multi-Object Tracking: A Survey: https://arxiv.org/abs/1907.12740
- FairMOT paper: https://arxiv.org/pdf/2004.01888.pdf
- FairMOT on GitHub: https://github.com/ifzhang/FairMOT




