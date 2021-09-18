
# From: https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping

import itertools
import random
import numpy as np


def find_nearest(array, value):
    value = int(value)
    array = np.asarray(array).astype(int)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def norm_arr(a):
    a = a-a.min()
    a = a/a.max()
    return a


def dist(a1, a2):
    return np.linalg.norm(a1-a2)


def dist_for_different_len(a1, a2):
    '''
    Compares player positions from NGS tracking with helmet positions
    from baseline helmet detection model.  

    Args:
        a1: 1-d np.array
            Player positions according to NGS tracking data.  
            [player0 position, player1 position, ...]

        a2: 1-d np.array
            Helmet positions from to baseline helmet detection model.
            [helmet0 position, helmet1 position, ...]

    Returns:
        min_dist: float
            A number indicating how closely a1 and a2 match 
            (Frobenius norm between them).  The lower the better match.

        min_detete_idx: tuple
            There may be more players in a1 than there are helmets in a2, 
            so randomly selected players from a1 are left out of a1 before it is 
            macthed and compared with a2 each time.  Over `max_iter` tries, 
            min_detete_idx are the indices of the players that have been
            left out for the closest match.
    '''
    max_iter = 2000

    assert len(a1) >= len(a2), f'{len(a1)}, {len(a2)}'
    len_diff = len(a1) - len(a2)  

    a2 = norm_arr(a2)

    if len_diff == 0:
        # Compare & match, 
        # if there're equal numbers of players and helmets to begin with.
        a1 = norm_arr(a1)
        return dist(a1, a2), ()
    else:
        min_dist = 10000
        min_detete_idx = None

        # All combinations of players that can be left out of a1 to keep
        # the number of players left equal to the number of helmets.
        del_list = list(itertools.combinations(range(len(a1)), len_diff))

        if len(del_list) > max_iter:
            del_list = random.sample(del_list, max_iter)

        # Leave out players from each combination, then match and compare with 
        # helmet positions, keeping track of the closest match.
        for detete_idx in del_list:
            this_a1 = np.delete(a1, detete_idx)
            this_a1 = norm_arr(this_a1)
            this_dist = dist(this_a1, a2)
            #print(len(a1), len(a2), this_dist)
            if min_dist > this_dist:
                min_dist = this_dist
                min_detete_idx = detete_idx

        return min_dist, min_detete_idx


def rotate_arr(u, t, deg=True):
    if deg == True:
        t = np.deg2rad(t)
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    return np.dot(R, u)


def dist_rot(tracking_df, a2):
    '''
    Rotating the NGS tracking frame bit by bit, matching and comparing
    the player positions with the helmet positions from the baseline
    helmet detection model, eventually assigning players to helmets at
    the closest match.

    Args:
        tracking_df: pd.DataFrame
            NGS tracking data for a given frame, containing importantly
            the players' (x, y) coordinates, in the tracking frame.
        a2: 1-d np.array
            Helmet positions from to baseline helmet detection model.
            [helmet0 position, helmet1 position, ...]

    Returns:
        min_dist: float
            A number indicating how closely  player positions from NGS tracking
            and helmet positions from baseline helmet detection model can match.
            The lower the better match. (Minimum Frobenius norm)
        players: 1d np.array
            Player numbers that are the best matches to the helmets
            from the baseline helmet detection model.
    '''
    tracking_df = tracking_df.sort_values('x')
    x = tracking_df['x']
    y = tracking_df['y']

    min_dist = 10000
    min_idx = None
    min_x = None

    dig_step = 3    # unit rotation in degrees
    dig_max = dig_step*10    
    for dig in range(-dig_max, dig_max+1, dig_step):
        arr = rotate_arr(np.array((x, y)), dig)    # rotate NGS tracking frame
        this_dist, this_idx = dist_for_different_len(np.sort(arr[0]), a2)
        if min_dist > this_dist:
            min_dist = this_dist
            min_idx = this_idx
            min_x = arr[0]
    tracking_df['x_rot'] = min_x   
    player_arr = tracking_df.sort_values('x_rot')['player'].values
    players = np.delete(player_arr, min_idx) 
    return min_dist, players


def mapping_df(video_frame, df, tracking, conf_thre=0.3):
    '''
    For a video frame, assign a player number to each helmet detected
    by the baseline helmet detection model. 

    Args:
        video_frame: str
            Video frame ID, consisting of gameKey, playID, view, frame 
            joined by '_'.
        df: pd.DataFrame
            Baseline helmet detection output for the frame.  Each row is a 
            helmet.  Columns include things like bounding box and detection
            confidence, etc.
        conf_thre: float
            Confidence threshold above which to keep detected helmet.

    Returns:
        tgt_df: pd.DataFrame
            Each row a helmet. Columns define bounding box and player number.
    '''
    gameKey, playID, view, frame = video_frame.split('_')
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)

    # Get NGS tracking data for this particular game, play, and frame
    this_tracking = tracking[(tracking['gameKey'] == gameKey) & (
        tracking['playID'] == playID)]
    est_frame = find_nearest(this_tracking.est_frame.values, frame)
    this_tracking = this_tracking[this_tracking['est_frame'] == est_frame]
    len_this_tracking = len(this_tracking)
    if view == 'Endzone': # Swap x and y positions if the view is from endzone
        this_tracking['x'], this_tracking['y'] = this_tracking['y'].copy(
        ), this_tracking['x'].copy()

    df = df[df['conf'] > conf_thre].copy() 
    # Truncate number of helmets to number of tracked players, keeping the 
    # most confident detections
    if len(df) > len_this_tracking:  
        df = df.tail(len_this_tracking)
    df['center_h_p'] = (df['left']+df['width']/2).astype(int)  # x-coordinate of helmet centre in camera frame
    df['center_h_m'] = (df['left']+df['width']/2).astype(int)*-1 # minus x-coordinate of helmet centre in camera frame
    df_p = df.sort_values('center_h_p').copy()
    df_m = df.sort_values('center_h_m').copy()
    a2_p = df_p['center_h_p'].values # Helmets sorted along x-axis
    a2_m = df_m['center_h_m'].values # Helmets sorted along x-axis in reverse order.

    # We will try matching both of these against the NGS tracking positions
    # because the camera can be either located on a home or away side of the pitch.
    # We will simply take the closest match of these two.
    min_dist_p, min_detete_idx_p = dist_rot(this_tracking, a2_p)
    min_dist_m, min_detete_idx_m = dist_rot(this_tracking, a2_m)
    if min_dist_p < min_dist_m:
        min_dist = min_dist_p
        min_detete_idx = min_detete_idx_p
        tgt_df = df_p
    else:
        min_dist = min_dist_m
        min_detete_idx = min_detete_idx_m
        tgt_df = df_m

    #print(video_frame, len(this_tracking), len(df), len(df[df['conf']>CONF_THRE]), this_tracking['x'].mean(), min_dist_p, min_dist_m, min_dist)
    tgt_df['label'] = min_detete_idx
    return tgt_df[['video_frame', 'left', 'width', 'top', 'height', 'label']]
