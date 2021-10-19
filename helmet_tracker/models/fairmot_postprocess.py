
import pandas as pd



def load_results_txt(pth='results.txt', video_width=1280, video_height=720):
    '''
    Load FairMOT tracking results for a video.
    '''
    fairmot_video_width = 1920
    fairmot_video_height = 1080
    x_scale = video_width / fairmot_video_width
    y_scale = video_height / fairmot_video_height

    columns = ['frame', 'id', 'x1', 'y1', 'w', 'h']

    df = pd.read_csv(pth, header=None,
                     usecols=range(len(columns)), names=columns)

    df[['x1', 'w']] = x_scale * df[['x1', 'w']]
    df[['y1', 'h']] = y_scale * df[['y1', 'h']]

    df = df[['frame', 'x1', 'y1', 'w', 'h', 'id']]

    df.rename(columns={'x1': 'left',
                       'y1': 'top',
                       'w': 'width',
                       'h': 'height',
                       'id': 'fairmot_cluster'}, inplace=True)

    for column in ['left', 'top', 'width', 'height']:
        df[column] = df[column].astype(int)

    return df
