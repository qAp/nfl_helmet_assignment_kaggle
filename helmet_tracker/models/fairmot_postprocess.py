
import pandas as pd
import cv2


def load_demo_txt(pth_fmot, video_width=1280, video_height=720):
    '''
    Load FairMOT inference output txt for a video.
    
    Args:
        pth_fmot (str): Path to txt file output by FairMOT's demo.py
        video_width (int): Actual video width.  Default: 1280
        video_height (int): Actual video height.  Default: 720
        
    Returns:
        df_fmot (pd.DataFrame): MOT tracks of helmets. Each row is a 
            helmet in some frame, with an MOT tracking id under column
            'fairmot_cluster'.
    '''
    columns = ['frame', 'id', 'x1', 'y1', 'w', 'h']
    df_fmot = pd.read_csv(pth_fmot, header=None, usecols=range(len(columns)),
                          names=columns)

    # Rescale output bboxes to match video's dimensions
    fmot_video_width = 1920
    fmot_video_height = 1080
    x_scale = video_width / fmot_video_width
    y_scale = video_height / fmot_video_height
    df_fmot[['x1', 'w']] = x_scale * df_fmot[['x1', 'w']]
    df_fmot[['y1', 'h']] = y_scale * df_fmot[['y1', 'h']]

    # Rename columns to be like for DeepSORT
    df_fmot.rename(columns={'id': 'fairmot_cluster',
                            'x1': 'left',
                            'y1': 'top',
                            'w': 'width',
                            'h': 'height'},
                   inplace=True)

    for c in ['left', 'top', 'width', 'height']:
        df_fmot[c] = df_fmot[c].astype(int)

    return df_fmot


def merge_hmap_fmot_bbox(hmap, fmot, drop_dupe_id=False):
    '''
    Merge helmet mapping bboxes and MOT bboxes for a single
    video frame.
    
    Args:
        hmap (pd.DataFrame): Each row a helmet, with columns such
            as 'video_frame', 'label', 'left', 'top', and 'bottom', etc.
        fmot (pd.DataFrame): Each row a helmet, with columns such
            as 'frame', 'fairmot_cluster', 'left', 'top', and 'width', etc.
        drop_dupe_id (bool): If True, repeated MOT ids (`fairmot_cluster`)
            are dropped but the one with 'top' that matches most closely
            with a helmet mapping box.  Otherwise, repeated MOT ids are
            kept.
            
    Returns:
        merged (pd.DataFrame): Each row a helmet with player number
            and MOT id, as well as bounding box properties.
    '''
    # Sort by 'left', then merge by matching the nearest left
    hmap.sort_values(['left'], axis=0, inplace=True)
    fmot.sort_values(['left'], axis=0, inplace=True)
    merged = pd.merge_asof(hmap, fmot, on='left',
                           direction='nearest', suffixes=('', '_fairmot'))

    if drop_dupe_id:
        # For duplicated fairmot_cluster, choose the best matched top.
        # Note this removes excess baseline helmets
        merged['dtop'] = (merged['top'] - merged['top_fairmot']).abs()
        merged.sort_values('dtop', inplace=True)
        merged = merged.groupby('fairmot_cluster').first().reset_index()

    return merged


def assign_player_to_track(df):
    '''
    Assign a player number to each MOT id occurence in a 
    video.
    
    Args:
        df (pd.DataFrame): Each row a helmet, with 'label'
            the player number according to helmet mapping and
            'fairmot_cluster' the MOT id according to FairMOT.
    Returns:
        df (pd.DataFrame): Same as input but with player number
            after incorporating FairMOT tracks added under 'label_fairmot'.
            'label_count_fairmot', the number of occurences of the most
            frequent helmet mapping player number for each MOT id's track.
    '''
    grpd = (df
            .groupby('fairmot_cluster')['label']
            .value_counts()
            .sort_values(ascending=False)
            .to_frame()
            .rename(columns={'label': 'label_count'})
            .reset_index()
            .groupby('fairmot_cluster')
            )

    cluster2player = grpd.first()['label'].to_dict()
    cluster2count = grpd.first()['label_count'].to_dict()

    df['label_fairmot'] = df['fairmot_cluster'].map(cluster2player)
    df['label_count_fairmot'] = df['fairmot_cluster'].map(cluster2count)

    return df


def fmot_postprocess_hmap(df_hmap, pth_fmot, pth_video=None):
    '''
    Postprocess helmet mapping results using FairMOT tracks,
     for a video.
     
    Args:
        df_hmap (pd.DataFrame): Helmet mapping results for a video.
        pth_fmot (str): File path FairMOT results for a video, a txt file.
        pth_video (str, None): File path to the video.  If supplied,
            the video is loaded and width and height are extracted,
            to be used to rescale the FairMOT bboxes.
            
    Returns: 
        df_tracks (pd.DataFrame): Helmets assigned with player numbers
            that taking into account both helmet mapping and FairMOT,
            for a video.
    '''
    if pth_video is not None:
        cap = cv2.VideoCapture(pth_video)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        df_fmot = load_demo_txt(pth_fmot, width, height)
    else:
        df_fmot = load_demo_txt(pth_fmot)

    frames = df_hmap['frame'].unique()
    df_tracks = []
    for frame in frames:
        hmap = df_hmap[df_hmap['frame'] == frame].copy()
        fmot = df_fmot[df_fmot['frame'] == frame].copy()

        merged = merge_hmap_fmot_bbox(hmap, fmot, drop_dupe_id=False)
        df_tracks.append(merged)

    df_tracks = pd.concat(df_tracks, axis=0)

    df_tracks = assign_player_to_track(df_tracks)

    return df_tracks
