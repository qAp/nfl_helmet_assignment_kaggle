'''
Based on: 
- https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping
'''

import itertools
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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


def dist_2d(x0, y0, x1, y1):
    '''
    Sum of distance scores in the x and y dimension.
    '''
    return dist(x0, x1) + dist(y0, y1)


def random_discard_rows(df, num_discard=0):
    '''
    Randomly discard a number of rows from dataframe.

    Args:
        df (pd.DataFrame): Dataframe a number of whose rows will be thrown
            away.
        num_discard (int): Number of rows to discard. Default: 0.
    Returns:
        idxs_discard (iter): Indexs of `df` at which the row is discarded.
        df_remain (pd.DataFrame): Leftover dataframe.
    '''
    df = df.reset_index(drop=True).copy()
    idxs_discard = random.sample(range(len(df)), num_discard)
    to_discard = df.index.isin(idxs_discard)
    df_remain = df[~to_discard].copy()
    return idxs_discard, df_remain


def sorted_norm_helmets_xy(df):
    '''
    Sort and normalise the x and y helmets coordinates 
    (camera reference frame).

    Args:
        df (pd.DataFrame): Dataframe containing helmets data.  Must
            contain columns 'left', 'top', 'width', and 'height'.
    Returns:
        x (np.array): Sorted and normalised helmets x-coordinates.
        y (np.array): Sorted and normalised helmets y-coordinates.   
    '''
    x = (df['left'] + 0.5 * df['width']).values
    x = np.sort(x)
    x = norm_arr(x)

    y = (df['top'] + df['height']).values  # the chin
    y = np.sort(y)
    y = norm_arr(y)

    return x, y


def sorted_norm_ngs_xy(df):
    '''
    Sort and normalise the x and y NGS coordinates 
    (NGS reference frame).

    Args:
        df (pd.DataFrame): Dataframe containing NGS data.  Must
            contain columns 'x' and 'y'.
    Returns:
        x (np.array): Sorted and normalised NGS x-coordinates.
        y (np.array): Sorted and normalised NGS y-coordinates.
    '''
    x = df['x'].values
    x = np.sort(x)
    x = norm_arr(x)

    y = df['y'].values
    y = np.sort(y)
    y = norm_arr(y)

    return x, y


def dist_2d_frame(df_hel, df_ngs, max_iter=2000):
    '''
    Compute positions difference between the camera and the NGS
     reference frames.

    Args:
        df_hel (pd.DataFrame): Helmets in the camera reference
            frame, including bounding boxes (top, left, width, height),
            and detection confidence.
        df_ngs (pd.DataFrame): Players in the NGS reference frame, 
            including player position (x, y).
        max_iter (int): Maximum number of times to randomly filter out
            players such that number of helmets and number of players
            are equal before position difference is computed.
    Returns:
        min_idxs_discard (iter, None): The indices in `df_ngs` at which
            the player is left out in obtaining `min_dist_score`.  `None`
            when the number of helmets and the number of players are 
            originally equal.
        min_dist_score (float): Minimum possible position difference
            between the camera and NGS reference frames.
    '''
    # If there're more helmets than they are NGS players, take the most 
    # confidently predicted helmets to match the number of NGS players.
    if len(df_hel) > len(df_ngs):
        df_hel = (df_hel
                  .sort_values('conf', ascending=False)
                  .iloc[:len(df_ngs)]
                  .copy())
    assert len(df_hel) <= len(df_ngs)

    x_hel, y_hel = sorted_norm_helmets_xy(df_hel)

    min_idxs_discard = None
    min_dist_score = 999
    if len(df_hel) == len(df_ngs):
        x_ngs, y_ngs = sorted_norm_ngs_xy(df_ngs)
        min_dist_score = dist_2d(x_hel, y_hel, x_ngs, y_ngs)
    else:
        num_discard = len(df_ngs) - len(df_hel)

        for _ in range(max_iter):
            idxs_discard, df_remain = random_discard_rows(df_ngs, num_discard)
            assert len(df_hel) == len(df_remain)
            x_ngs, y_ngs = sorted_norm_ngs_xy(df_remain)
            dist_score = dist_2d(x_hel, y_hel, x_ngs, y_ngs)

            if dist_score < min_dist_score:
                min_idxs_discard = idxs_discard
                min_dist_score = dist_score
    
    return min_idxs_discard, min_dist_score


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


def rotate_dataframe(df, t=0, deg=True):
    '''
    Rotate the x and y coordinates saved in a dataframe.

    Args:
        df (pd.DataFrame): Must contain columns 'x' and 'y', the x-coordinates
            and the y-coordinates of a set of positions.
        t (float): Angle of rotation.  Default: 0.
        deg (bool): `t` is in degress if True, otherwise in radians.  Default: True.

    Returns:
        df_theta (pd.DataFrame): Same as df, but with 'x' and 'y' changed after
            rotation by angle `t`.
    '''
    df_theta = df.copy()
    xy_values = df_theta[['x', 'y']].values.T  # shape (2, number of locations)
    df_theta['x'], df_theta['y'] = rotate_arr(xy_values, t, deg)
    return df_theta



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
        tracking (pd.DataFrame): NGS tracking data for all video and frames.  Usually
            loaded from competition csv file.
        conf_thre (float): Confidence threshold above which to keep detected helmet.
            Default: 0.3.

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


def dist_rot_2d(df_hel, df_ngs, ts, t_init=0):
    if t_init % 360 != 0:
        df_ngs = rotate_dataframe(df_ngs, t_init)
    else: 
        df_ngs = df_ngs.copy()

    score_min = 10_000
    for t in ts:
        df_t = rotate_dataframe(df_ngs, t=t, deg=True)
        idxs_discard, dist_score = dist_2d_frame(df_hel, df_t, max_iter=1000)

        if dist_score < score_min:
            score_min = dist_score

            if idxs_discard is None:
                df_min = df_t.copy()
            else:
                # NGS players with some discarded
                df_min = df_t.reset_index(drop=True).copy()
                to_discard = df_min.index.isin(idxs_discard)
                df_min = df_min[~to_discard]

    # Sort the x-cooridnates and assign players to helmets
    assert len(df_min) == len(df_hel)
    label = df_min.sort_values('x')['player'].values
    df_tgt = df_hel.copy()
    df_tgt['x'] = df_tgt['left'] + 0.5 * df_tgt['width']
    df_tgt.sort_values('x', axis=0, inplace=True)
    df_tgt['label'] = label

    return score_min, df_tgt

    

def mapping_df_2d(video_frame, df, tracking, conf_thre=0.3):
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
        tracking (pd.DataFrame): NGS tracking data for all video and frames.  Usually
            loaded from competition csv file.
        conf_thre (float): Confidence threshold above which to keep detected helmet.
            Default: 0.3.

    Returns:
        tgt_df: pd.DataFrame
            Each row a helmet. Columns define bounding box and player number.
    '''
    gameKey, playID, view, frame = video_frame.split('_')
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)

    # Get NGS players
    df_ngs = tracking[(tracking['gameKey'] == gameKey) & (
        tracking['playID'] == playID)]
    est_frame = find_nearest(df_ngs.est_frame.values, frame)
    df_ngs = df_ngs[df_ngs['est_frame'] == est_frame]

    # Get helmets
    df_hel = df[df['conf'] > conf_thre].copy()
    if len(df_hel) > len(df_ngs):
        df_hel = df_hel.tail(len(df_ngs))

    # Define which rotation angles to try
    tmax = 30
    num_t = 20
    ts = np.linspace(-tmax, tmax, num_t)

    # Compute min dist for 2 of the possible pitch sides
    t_init = 0 if view == 'Sideline' else 90
    min_dist_p, tgt_df_p = dist_rot_2d(df_hel, df_ngs, ts, t_init=t_init)
    min_dist_m, tgt_df_m = dist_rot_2d(df_hel, df_ngs, ts, t_init=t_init + 180)

    tgt_df = tgt_df_p if min_dist_p < min_dist_m else tgt_df_m

    return tgt_df[['video_frame', 'left', 'width', 'top', 'height', 'label']]
    




