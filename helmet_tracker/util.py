

import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pylab as plt
import matplotlib.patches as patches
from IPython.display import Video, display
import subprocess
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def check_submission(sub):
    """
    Checks that the submission meets all the requirements.

    1. No more than 22 Boxes per frame.
    2. Only one label prediction per video/frame
    3. No duplicate boxes per frame.

    Returns:
        True -> Passed the tests
        False -> Failed the test

    Notes:
    This function is taken from: 
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
    """
    # Maximum of 22 boxes per frame.
    max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
    if max_box_per_frame > 22:
        print("Has more than 22 boxes in a single frame")
        return False
    # Only one label allowed per frame.
    has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
    if has_duplicate_labels:
        print("Has duplicate labels")
        return False
    # Check for unique boxes
    has_duplicate_boxes = (
        sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
    )
    if has_duplicate_boxes:
        print("Has duplicate boxes")
        return False
    return True


class NFLAssignmentScorer:
    def __init__(
        self,
        labels_df: pd.DataFrame = None,
        labels_csv="train_labels.csv",
        check_constraints=True,
        weight_col="isDefinitiveImpact",
        impact_weight=1000,
        iou_threshold=0.35,
        remove_sideline=True,
    ):
        """
        Helper class for grading submissions in the
        2021 Kaggle Competition for helmet assignment.
        Version 1.0
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide

        Use:
        ```
        scorer = NFLAssignmentScorer(labels)
        scorer.score(submission_df)

        or

        scorer = NFLAssignmentScorer(labels_csv='labels.csv')
        scorer.score(submission_df)
        ```

        Args:
            labels_df (pd.DataFrame, optional):
                Dataframe containing theground truth label boxes.
            labels_csv (str, optional): CSV of the ground truth label.
            check_constraints (bool, optional): Tell the scorer if it
                should check the submission file to meet the competition
                constraints. Defaults to True.
            weight_col (str, optional):
                Column in the labels DataFrame used to applying the scoring
                weight.
            impact_weight (int, optional):
                The weight applied to impacts in the scoring metrics.
                Defaults to 1000.
            iou_threshold (float, optional):
                The minimum IoU allowed to correctly pair a ground truth box
                with a label. Defaults to 0.35.
            remove_sideline (bool, optional):
                Remove slideline players from the labels DataFrame
                before scoring.

        Notes:
        This function is taken from: 
            https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
        """
        if labels_df is None:
            # Read label from CSV
            if labels_csv is None:
                raise Exception("labels_df or labels_csv must be provided")
            else:
                self.labels_df = pd.read_csv(labels_csv)
        else:
            self.labels_df = labels_df.copy()
        if remove_sideline:
            self.labels_df = (
                self.labels_df.query("isSidelinePlayer == False")
                .reset_index(drop=True)
                .copy()
            )
        self.impact_weight = impact_weight
        self.check_constraints = check_constraints
        self.weight_col = weight_col
        self.iou_threshold = iou_threshold

    def check_submission(self, sub):
        """
        Checks that the submission meets all the requirements.

        1. No more than 22 Boxes per frame.
        2. Only one label prediction per video/frame
        3. No duplicate boxes per frame.

        Args:
            sub : submission dataframe.

        Returns:
            True -> Passed the tests
            False -> Failed the test
        """
        # Maximum of 22 boxes per frame.
        max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
        if max_box_per_frame > 22:
            print("Has more than 22 boxes in a single frame")
            return False
        # Only one label allowed per frame.
        has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
        if has_duplicate_labels:
            print("Has duplicate labels")
            return False
        # Check for unique boxes
        has_duplicate_boxes = (
            sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
        )
        if has_duplicate_boxes:
            print("Has duplicate boxes")
            return False
        return True

    def add_xy(self, df):
        """
        Adds `x1`, `x2`, `y1`, and `y2` columns necessary for computing IoU.

        Note - for pixel math, 0,0 is the top-left corner so box orientation
        defined as right and down (height)
        """

        df["x1"] = df["left"]
        df["x2"] = df["left"] + df["width"]
        df["y1"] = df["top"]
        df["y2"] = df["top"] + df["height"]
        return df

    def merge_sub_labels(self, sub, labels, weight_col="isDefinitiveImpact"):
        """
        Perform an outer join between submission and label.
        Creates a `sub_label` dataframe which stores the matched label for each submission box.
        Ground truth values are given the `_gt` suffix, submission values are given `_sub` suffix.
        """
        sub = sub.copy()
        labels = labels.copy()

        sub = self.add_xy(sub)
        labels = self.add_xy(labels)

        base_columns = [
            "label",
            "video_frame",
            "x1",
            "x2",
            "y1",
            "y2",
            "left",
            "width",
            "top",
            "height",
        ]

        sub_labels = sub[base_columns].merge(
            labels[base_columns + [weight_col]],
            on=["video_frame"],
            how="right",
            suffixes=("_sub", "_gt"),
        )
        return sub_labels

    def get_iou_df(self, df):
        """
        This function computes the IOU of submission (sub)
        bounding boxes against the ground truth boxes (gt).
        """
        df = df.copy()

        # 1. get the coordinate of inters
        df["ixmin"] = df[["x1_sub", "x1_gt"]].max(axis=1)
        df["ixmax"] = df[["x2_sub", "x2_gt"]].min(axis=1)
        df["iymin"] = df[["y1_sub", "y1_gt"]].max(axis=1)
        df["iymax"] = df[["y2_sub", "y2_gt"]].min(axis=1)

        df["iw"] = np.maximum(df["ixmax"] - df["ixmin"] + 1, 0.0)
        df["ih"] = np.maximum(df["iymax"] - df["iymin"] + 1, 0.0)

        # 2. calculate the area of inters
        df["inters"] = df["iw"] * df["ih"]

        # 3. calculate the area of union
        df["uni"] = (
            (df["x2_sub"] - df["x1_sub"] + 1) *
            (df["y2_sub"] - df["y1_sub"] + 1)
            + (df["x2_gt"] - df["x1_gt"] + 1) * (df["y2_gt"] - df["y1_gt"] + 1)
            - df["inters"]
        )
        # print(uni)
        # 4. calculate the overlaps between pred_box and gt_box
        df["iou"] = df["inters"] / df["uni"]

        return df.drop(
            ["ixmin", "ixmax", "iymin", "iymax", "iw", "ih", "inters", "uni"], axis=1
        )

    def filter_to_top_label_match(self, sub_labels):
        """
        Ensures ground truth boxes are only linked to the box
        in the submission file with the highest IoU.
        """
        return (
            sub_labels.sort_values("iou", ascending=False)
            .groupby(["video_frame", "label_gt"])
            .first()
            .reset_index()
        )

    def add_isCorrect_col(self, sub_labels):
        """
        Adds True/False column if the ground truth label
        and submission label are identical
        """
        sub_labels["isCorrect"] = (
            sub_labels["label_gt"] == sub_labels["label_sub"]
        ) & (sub_labels["iou"] >= self.iou_threshold)
        return sub_labels

    def calculate_metric_weighted(
        self, sub_labels, weight_col="isDefinitiveImpact", weight=1000
    ):
        """
        Calculates weighted accuracy score metric.
        """
        sub_labels["weight"] = sub_labels.apply(
            lambda x: weight if x[weight_col] else 1, axis=1
        )
        y_pred = sub_labels["isCorrect"].values
        y_true = np.ones_like(y_pred)
        weight = sub_labels["weight"]
        return accuracy_score(y_true, y_pred, sample_weight=weight)

    def score(self, sub, labels_df=None, drop_extra_cols=True):
        """
        Scores the submission file against the labels.

        Returns the evaluation metric score for the helmet
        assignment kaggle competition.

        If `check_constraints` is set to True, will return -999 if the
            submission fails one of the submission constraints.
        """
        if labels_df is None:
            labels_df = self.labels_df.copy()

        if self.check_constraints:
            if not self.check_submission(sub):
                return -999
        sub_labels = self.merge_sub_labels(sub, labels_df, self.weight_col)
        sub_labels = self.get_iou_df(sub_labels).copy()
        sub_labels = self.filter_to_top_label_match(sub_labels).copy()
        sub_labels = self.add_isCorrect_col(sub_labels)
        score = self.calculate_metric_weighted(
            sub_labels, self.weight_col, self.impact_weight
        )
        # Keep `sub_labels for review`
        if drop_extra_cols:
            drop_cols = [
                "x1_sub",
                "x2_sub",
                "y1_sub",
                "y2_sub",
                "x1_gt",
                "x2_gt",
                "y1_gt",
                "y2_gt",
            ]
            sub_labels = sub_labels.drop(drop_cols, axis=1)
        self.sub_labels = sub_labels
        return score


def video_with_baseline_boxes(
    video_path: str, baseline_boxes: pd.DataFrame, gt_labels: pd.DataFrame, verbose=True
) -> str:
    """
    Annotates a video with both the baseline model boxes and ground truth boxes.
    Baseline model prediction confidence is also displayed.

    Notes:
        This function is taken from: 
            https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
    """
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (0, 0, 0)  # Black
    BASELINE_COLOR = (255, 255, 255)  # White
    IMPACT_COLOR = (0, 0, 255)  # Red
    video_name = os.path.basename(video_path).replace(".mp4", "")
    if verbose:
        print(f"Running for {video_name}")
    baseline_boxes = baseline_boxes.copy()
    gt_labels = gt_labels.copy()

    baseline_boxes["video"] = (
        baseline_boxes["video_frame"].str.split("_").str[:3].str.join("_")
    )
    gt_labels["video"] = gt_labels["video_frame"].str.split(
        "_").str[:3].str.join("_")
    baseline_boxes["frame"] = (
        baseline_boxes["video_frame"].str.split("_").str[-1].astype("int")
    )
    gt_labels["frame"] = gt_labels["video_frame"].str.split(
        "_").str[-1].astype("int")

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "labeled_" + video_name + ".mp4"
    tmp_output_path = "tmp_" + output_path
    output_video = cv2.VideoWriter(
        tmp_output_path, cv2.VideoWriter_fourcc(
            *VIDEO_CODEC), fps, (width, height)
    )
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        # We need to add 1 to the frame count to match the label frame index
        # that starts at 1
        frame += 1

        # Let's add a frame index to the video so we can track where we are
        img_name = f"{video_name}_frame{frame}"
        cv2.putText(
            img,
            img_name,
            (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            HELMET_COLOR,
            thickness=2,
        )

        # Now, add the boxes
        boxes = baseline_boxes.query(
            "video == @video_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            return
        for box in boxes.itertuples(index=False):
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                BASELINE_COLOR,
                thickness=1,
            )
            cv2.putText(
                img,
                f"{box.conf:0.2}",
                (box.left, max(0, box.top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BASELINE_COLOR,
                thickness=1,
            )

        boxes = gt_labels.query("video == @video_name and frame == @frame")
        if len(boxes) == 0:
            print("Boxes incorrect")
            return
        for box in boxes.itertuples(index=False):
            # Filter for definitive head impacts and turn labels red
            if box.isDefinitiveImpact == True:
                color, thickness = IMPACT_COLOR, 3
            else:
                color, thickness = HELMET_COLOR, 1
            cv2.rectangle(
                img,
                (box.left, box.top),
                (box.left + box.width, box.top + box.height),
                color,
                thickness=thickness,
            )
            cv2.putText(
                img,
                box.label,
                (box.left + 1, max(0, box.top - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
            )

        output_video.write(img)
    output_video.release()
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            tmp_output_path,
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-vcodec",
            "libx264",
            output_path,
        ]
    )
    os.remove(tmp_output_path)

    return output_path


def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.

    Notes:
    This function is taken from: 
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace(
        "H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype(
        "timedelta64[ms]"
    ) / 1_000
    # Estimated video frame
    tracks["est_frame"] = (
        ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    )
    return tracks


def create_football_field(
    linenumbers=True,
    endzones=True,
    highlight_line=False,
    highlight_line_number=50,
    highlighted_name="Line of Scrimmage",
    fifty_is_los=False,
    figsize=(12, 6.33),
    field_color="lightgreen",
    ez_color='forestgreen',
    ax=None,
):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.

    Notes:
    This function is taken from: 
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
    """
    rect = patches.Rectangle(
        (0, 0),
        120,
        53.3,
        linewidth=0.1,
        edgecolor="r",
        facecolor=field_color,
        zorder=0,
    )

    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='black')

    if fifty_is_los:
        ax.plot([60, 60], [0, 53.3], color="gold")
        ax.text(62, 50, "<- Player Yardline at Snap", color="gold")
    # Endzones
    if endzones:
        ez1 = patches.Rectangle(
            (0, 0),
            10,
            53.3,
            linewidth=0.1,
            edgecolor="black",
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ez2 = patches.Rectangle(
            (110, 0),
            120,
            53.3,
            linewidth=0.1,
            edgecolor="black",
            facecolor=ez_color,
            alpha=0.6,
            zorder=0,
        )
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    ax.axis("off")
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(
                x,
                5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color="black",
            )
            ax.text(
                x - 0.95,
                53.3 - 5,
                str(numb - 10),
                horizontalalignment="center",
                fontsize=20,  # fontname='Arial',
                color="black",
                rotation=180,
            )
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color="black")
        ax.plot([x, x], [53.0, 52.5], color="black")
        ax.plot([x, x], [22.91, 23.57], color="black")
        ax.plot([x, x], [29.73, 30.39], color="black")

    if highlight_line:
        hl = highlight_line_number + 10
        ax.plot([hl, hl], [0, 53.3], color="yellow")
        ax.text(hl + 2, 50, "<- {}".format(highlighted_name), color="yellow")

    border = patches.Rectangle(
        (-5, -5),
        120 + 10,
        53.3 + 10,
        linewidth=0.1,
        edgecolor="orange",
        facecolor="white",
        alpha=0,
        zorder=0,
    )
    ax.add_patch(border)
    ax.set_xlim((0, 120))
    ax.set_ylim((0, 53.3))
    return ax


def add_plotly_field(fig):
    # Reference https://www.kaggle.com/ammarnassanalhajali/nfl-big-data-bowl-2021-animating-players
    fig.update_traces(marker_size=20)

    fig.update_layout(paper_bgcolor='#29a500', plot_bgcolor='#29a500', font_color='white',
                      width=800,
                      height=600,
                      title="",

                      xaxis=dict(
                          nticks=10,
                          title="",
                          visible=False
                      ),

                      yaxis=dict(
                          scaleanchor="x",
                          title="Temp",
                          visible=False
                      ),
                      showlegend=True,

                      annotations=[
                          dict(
                              x=-5,
                              y=26.65,
                              xref="x",
                              yref="y",
                              text="ENDZONE",
                              font=dict(size=16, color="#e9ece7"),
                              align='center',
                              showarrow=False,
                              yanchor='middle',
                              textangle=-90
                          ),
                          dict(
                              x=105,
                              y=26.65,
                              xref="x",
                              yref="y",
                              text="ENDZONE",
                              font=dict(size=16, color="#e9ece7"),
                              align='center',
                              showarrow=False,
                              yanchor='middle',
                              textangle=90
                          )],
                      legend=dict(
                          traceorder="normal",
                          font=dict(family="sans-serif", size=12),
                          title="",
                          orientation="h",
                          yanchor="bottom",
                          y=1.00,
                          xanchor="center",
                          x=0.5
                      ),
                      )
    ####################################################

    fig.add_shape(type="rect", x0=-10, x1=0,  y0=0, y1=53.3,
                  line=dict(color="#c8ddc0", width=3), fillcolor="#217b00", layer="below")
    fig.add_shape(type="rect", x0=100, x1=110, y0=0, y1=53.3, line=dict(
        color="#c8ddc0", width=3), fillcolor="#217b00", layer="below")
    for x in range(0, 100, 10):
        fig.add_shape(type="rect", x0=x,   x1=x+10, y0=0, y1=53.3,
                      line=dict(color="#c8ddc0", width=3), fillcolor="#29a500", layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=1, x1=x, y1=2,
                      line=dict(color="#c8ddc0", width=2), layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=51.3, x1=x, y1=52.3,
                      line=dict(color="#c8ddc0", width=2), layer="below")

    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=20.0, x1=x, y1=21,
                      line=dict(color="#c8ddc0", width=2), layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line", x0=x, y0=32.3, x1=x, y1=33.3,
                      line=dict(color="#c8ddc0", width=2), layer="below")

    fig.add_trace(go.Scatter(
        x=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98], y=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        text=["G", "1 0", "2 0", "3 0", "4 0",
              "5 0", "4 0", "3 0", "2 0", "1 0", "G"],
        mode="text",
        textfont=dict(size=20, family="Arail"),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=[2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98], y=[48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3, 48.3],
        text=["G", "1 0", "2 0", "3 0", "4 0",
              "5 0", "4 0", "3 0", "2 0", "1 0", "G"],
        mode="text",
        textfont=dict(size=20, family="Arail"),
        showlegend=False,
    ))

    return fig


def random_label_submission(helmets, tracks):
    """
    Creates a baseline submission with randomly assigned helmets
    based on the top 22 most confident baseline helmet boxes for
    a frame.

    Notes:
    This function is taken from: 
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
    """
    # Take up to 22 helmets per frame based on confidence:
    helm_22 = (
        helmets.sort_values("conf", ascending=False)
        .groupby("video_frame")
        .head(22)
        .sort_values("video_frame")
        .reset_index(drop=True)
        .copy()
    )
    # Identify player label choices for each game_play
    game_play_choices = tracks.groupby(
        ["game_play"])["player"].unique().to_dict()
    # Loop through frames and randomly assign boxes
    ds = []
    helm_22["label"] = np.nan
    for video_frame, data in helm_22.groupby("video_frame"):
        game_play = video_frame[:12]
        choices = game_play_choices[game_play]
        np.random.shuffle(choices)
        data["label"] = choices[: len(data)]
        ds.append(data)
    submission = pd.concat(ds)
    return submission
