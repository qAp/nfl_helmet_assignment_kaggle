'''
Based on: 
- https://www.kaggle.com/its7171/nfl-baseline-simple-helmet-mapping
'''
import time
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
        print(f'Need to discard {num_discard} NGS players')

        for _ in range(max_iter):
            t0 = time.time()
            idxs_discard, df_remain = random_discard_rows(df_ngs, num_discard)
            print(f'random_discard_rows: {time.time() - t0:.6f} s')

            t0 = time.time()
            assert len(df_hel) == len(df_remain)
            print(f'assert: {time.time() - t0:.6f} s')

            t0 = time.time()
            x_ngs, y_ngs = sorted_norm_ngs_xy(df_remain)
            print(f'sorted_norm_ngs_xy: {time.time() - t0:.6f} s')

            t0 = time.time()
            dist_score = dist_2d(x_hel, y_hel, x_ngs, y_ngs)
            print(f'dist_2d: {time.time() - t0:.6f} s')

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
        print(f'(1d) Need to discard NGS players')
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
            t0 = time.time()
            this_a1 = np.delete(a1, detete_idx)
            print(f'np.delete: {time.time() - t0:.6f} s')

            t0 = time.time()
            this_a1 = norm_arr(this_a1)
            print(f'norm_arr: {time.time() - t0:.6f} s')

            t0 = time.time()
            this_dist = dist(this_a1, a2)
            print(f'dist: {time.time() - t0:.6f} s')

            #print(len(a1), len(a2), this_dist)
            if min_dist > this_dist:
                min_dist = this_dist
                min_detete_idx = detete_idx


        return min_dist, min_detete_idx


def dist_for_different_len_2d(xy_rot, xy_hel, max_iter=1000):
    '''
    Compute distance score between 2 sets of points, using their
    (x, y) coordinates.

    Args:
        xy_rot (np.array): Each row is an NGS player.  Columns 0 and 1
            are the x and y coordinates, respectively.  They don't need
            to be sorted, nor normalised.
        xy_hel (np.array): Each row a helmet.  Columns 0 and 1
            are the x and y coordinates, respectively, sorted in ascending
            x, and x and y are separately normalised.
        max_iter (int): Maximum number of combinations of NGS players to
            leave out, should there be more NGS players than there are
            helmets in `xy_hel`.  Default: 1000
    
    Returns:
        min_idxs_discard (list, tuple): Indices, corresponding to `xy_ngs`,
            of any players discarded, in order to obtain the minimum distance
            score.
        min_dist_score (float): Minimum distance score.
    '''
    assert len(xy_rot) >= len(xy_hel)
    num_discard = len(xy_rot) - len(xy_hel)

    if num_discard == 0:
        xy_rot = xy_rot[xy_rot[:, 0].argsort()]
        xy_rot -= xy_rot.min(axis=0, keepdims=True)
        xy_rot /= xy_rot.max(axis=0, keepdims=True)

        min_dist_score = dist(xy_hel, xy_rot)
        min_idxs_discard = ()

    else:
        min_dist_score = 10_000
        min_idxs_discard = None
        for _ in range(max_iter):
            idxs_discard = random.sample(range(len(xy_rot)), num_discard)
            xy = np.delete(xy_rot, idxs_discard, axis=0)

            xy = xy[xy[:, 0].argsort()]
            xy -= xy.min(axis=0, keepdims=True)
            xy /= xy.max(axis=0, keepdims=True)

            dist_score = dist(xy_hel, xy)

            if dist_score < min_dist_score:
                min_dist_score = dist_score
                min_idxs_discard = idxs_discard

    return min_idxs_discard, min_dist_score


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


def dist_rot_2d(df_ngs, xy_hel, t_init=0):
    '''
    Rotate NGS reference frame first by `t_init` degrees, then rotate bit
    by bit, computing and recording the minimum distance score.

    Args:
        df_ngs (pd.DataFrame): Each row is an NGS player, with columns 
            'x' and 'y', the x and y coordinates of the player.  
            Column 'player' contains the player lables. 
            e.g. 'H12', 'V32', etc.
        xy_hel (np.array): Each row is a helmet.  Column 0 and 1 are the
            x and y coordinates of the helmets, respectively. Helmets
            should already be sorted in ascending x-coordinate.  Both
            x and y coorindates should already be normalised separately.
        t_init (float): Initial angle of rotation of NGS reference frame
            in degrees.

    Returns:
        min_dist_score (float): The minimum possible distance score between 
            the camera reference frame and the NGS reference frame. 
        players (np.array): Player labels corresponding to the helmets in
            `xy_hel`.
    '''
    df_ngs = df_ngs.copy()
    xy_ngs = df_ngs[['x', 'y']].values

    if (t_init % 360) != 0:
        xy_ngs = rotate_arr(xy_ngs.T, t_init).T

    tmax = 30
    num_t = 20
    ts = np.linspace(-tmax, tmax, num_t)

    min_dist_score = 10_000
    min_idxs_discard = None
    min_xy_rot = None
    for t in ts:
        xy_rot = rotate_arr(xy_ngs.T, t).T

        (idxs_discard,
         dist_score) = _dist_for_different_len_2d(xy_rot, xy_hel)

        if dist_score < min_dist_score:
            min_dist_score = dist_score
            min_idxs_discard = idxs_discard
            min_xy_rot = xy_rot

    df_ngs[['x_rot', 'y_rot']] = min_xy_rot
    if len(min_idxs_discard) > 0:
        df_ngs.drop(df_ngs.index[min_idxs_discard], inplace=True)
    players = df_ngs.sort_values('x_rot')['player'].values
    return min_dist_score, players


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


def mapping_df_2d(video_frame, df, tracking, conf_thre=0.3):
    '''
    Map NGS players to helmet bounding boxes.

    Args:
        video_frame (str): Consisting of game ID, play ID, and frame.
        df (pd.DataFrame): Each row a helmet, with columns such as 
            'left', 'top', etc. for bounding boxes, 'video' for video
            ID, and 'frame' for frame number.
        tracking (pd.DataFrame): NGS tracking data.  Each row a player.
            Column 'x' and 'y' for player positions.  Column 'player'
            for player numbers.
        conf_thre (float): Helmet detection threshold below which helmets
            are not considered for the mapping process.

    Returns:
        df_tgt (pd.DataFrame): Helmets for `video_frame`.  Each row a helmet.
            Columns 'left', 'top', 'width', 'height', and 'label', where
            'label' are player numbers, such as 'H23', and 'V15', etc.
    '''
    gameKey, playID, view, frame = video_frame.split('_')
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)

    df_ngs = tracking.query('gameKey==@gameKey and playID==@playID')
    nearest_est_frame = find_nearest(df_ngs['est_frame'].values, frame)
    df_ngs = df_ngs.query('est_frame==@nearest_est_frame')

    # Get helmets for selected video and frame
    df_hel = df.query('video==@video and frame==@frame')
    df_hel = df_hel[df_hel['conf'] > conf_thre].copy()
    if len(df_hel) > len(df_ngs):
        df_hel = df_hel.tail(len(df_ngs))

    # Helmet centres
    df_hel['helmet_center_x'] = (df_hel['left'] + df_hel['width'] / 2)
    df_hel['helmet_center_y'] = (df_hel['top'] + df_hel['height'] / 2)

    # Sort helmets by their x-coordinate
    df_hel.sort_values('helmet_center_x', inplace=True)

    # Helmets' (x, y) coordinates
    xy_hel = df_hel[['helmet_center_x', 'helmet_center_y']].values
    # Flip y-axis to get same right-handedness as NGS reference frame
    xy_hel[:, 1] *= -1
    xy_hel -= xy_hel.min(axis=0, keepdims=True)
    xy_hel /= xy_hel.max(axis=0, keepdims=True)

    t_init = 0 if view == 'Sideline' else 90

    dist_p, players_p = _dist_rot_2d(df_ngs, xy_hel, t_init)
    dist_m, players_m = _dist_rot_2d(df_ngs, xy_hel, t_init + 180)

    if dist_p < dist_m:
        min_dist = dist_p
        min_players = players_p
    else:
        min_dist = dist_m
        min_players = players_m

    df_tgt = df_hel.copy()
    df_tgt['label'] = min_players
    return df_tgt[['video_frame', 'left', 'width', 'top', 'height', 'label']]

    




