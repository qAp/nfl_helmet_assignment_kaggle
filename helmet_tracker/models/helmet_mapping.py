
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


max_iter = 2000


def dist_for_different_len(a1, a2):
    assert len(a1) >= len(a2), f'{len(a1)}, {len(a2)}'
    len_diff = len(a1) - len(a2)
    a2 = norm_arr(a2)
    if len_diff == 0:
        a1 = norm_arr(a1)
        return dist(a1, a2), ()
    else:
        min_dist = 10000
        min_detete_idx = None
        cnt = 0
        del_list = list(itertools.combinations(range(len(a1)), len_diff))
        if len(del_list) > max_iter:
            del_list = random.sample(del_list, max_iter)
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
    tracking_df = tracking_df.sort_values('x')
    x = tracking_df['x']
    y = tracking_df['y']
    min_dist = 10000
    min_idx = None
    min_x = None
    dig_step = 3
    dig_max = dig_step*10
    for dig in range(-dig_max, dig_max+1, dig_step):
        arr = rotate_arr(np.array((x, y)), dig)
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

    gameKey, playID, view, frame = video_frame.split('_')
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)

    this_tracking = tracking[(tracking['gameKey'] == gameKey) & (
        tracking['playID'] == playID)]
    est_frame = find_nearest(this_tracking.est_frame.values, frame)
    this_tracking = this_tracking[this_tracking['est_frame'] == est_frame]
    len_this_tracking = len(this_tracking)
    if view == 'Endzone':
        this_tracking['x'], this_tracking['y'] = this_tracking['y'].copy(
        ), this_tracking['x'].copy()

    df = df[df['conf'] > conf_thre].copy()
    if len(df) > len_this_tracking:
        df = df.tail(len_this_tracking)
    df['center_h_p'] = (df['left']+df['width']/2).astype(int)
    df['center_h_m'] = (df['left']+df['width']/2).astype(int)*-1
    df_p = df.sort_values('center_h_p').copy()
    df_m = df.sort_values('center_h_m').copy()
    a2_p = df_p['center_h_p'].values
    a2_m = df_m['center_h_m'].values

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
