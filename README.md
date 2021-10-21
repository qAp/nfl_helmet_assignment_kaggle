# Code for NFL Health & Safety - Helmet Assignment Kaggle

## TODOs
- [] Tune helmet tracking
- [] Tune FairMOT


## Filtering out excess NGS positions
When the angle of rotation is good, the filtering algorithm appears to filter out the correct players that are out of view in the camera:
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/85662dac99363341185d1edf4014c86fe0ac791b/images/ngs_players_filtering/good_theta_good_filtering.png" width="800">

When the angle of rotation isn't good, the incorrect players may be left out:
<img src="https://github.com/qAp/nfl_helmet_assignment_kaggle/blob/85662dac99363341185d1edf4014c86fe0ac791b/images/ngs_players_filtering/bad_theta_bad_filtering.png" width="800">

## References
- https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
- https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping
- https://www.kaggle.com/robikscube/helper-code-helmet-mapping-deepsort
- SORT paper: https://arxiv.org/pdf/1602.00763.pdf
- DeepSORT paper: https://arxiv.org/pdf/1703.07402.pdf
- Deep Learning in Video Multi-Object Tracking: A Survey: https://arxiv.org/abs/1907.12740
- FairMOT paper: https://arxiv.org/pdf/2004.01888.pdf
- FairMOT on GitHub: https://github.com/ifzhang/FairMOT




