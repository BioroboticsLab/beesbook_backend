import os
import uuid
from subprocess import check_output

import matplotlib
import shutil

matplotlib.use('Agg')  # need to be executed before pyplot import, deactivates showing of plot in ipython
import matplotlib.pyplot as plt
import numpy as np


from plotter import utils

GPU = False
if GPU:
    from . import config_gpu as config
else:
    from . import config


def extract_frames(framecontainer):
    """
    Extracts all frame-images of the corresponding video file of a FrameContainer.

    Args:
        framecontainer (FrameContainer): The FrameContainer which represents the video file from which the frames
         should be extracted

    Returns: Directory path of the extracted images

    """
    video_name = framecontainer.video_name
    video_path = framecontainer.video_path
    output_path = f'/tmp/{video_name}'

    # check if files already exist
    if os.path.exists(output_path):
        from plotter.models import Frame
        if len(os.listdir(output_path)) == Frame.objects.filter(fc=framecontainer).count():
            return output_path

    os.makedirs(output_path, exist_ok=True)
    cmd = config.ffmpeg_extract_all_frames.format(video_path=video_path, output_path=output_path)
    output = check_output(cmd, shell=True)
    print('output:', output)

    return output_path


def extract_single_frame(frame):
    """
    Extracts the image to a `Frame`-object.
    Args:
        frame (Frame): The frame which should be extracted.

    Returns: The path to the image.

    """
    video_name = frame.fc.video_name
    video_path = frame.fc.video_path

    output_path = f'/tmp/{video_name}/{frame.index:04}.png'

    if not os.path.exists(output_path):
        cmd = config.ffmpeg_extract_single_frame.format(
            video_path=video_path,
            frame_index=frame.index,
            output_path=output_path
        )
        output = check_output(cmd, shell=True)
        print('output:', output)

    return output_path


# todo refactor, make it use django objects
def extract_video_subset(video_path, left_frame_idx, right_frame_idx):
    number_of_frames = right_frame_idx - left_frame_idx
    name = utils.get_filename(video_path)

    output_path = f'/tmp/{name}-{left_frame_idx}-{right_frame_idx}.mp4'

    if not os.path.exists(output_path):
        cmd = config.ffmpeg_video.format(**locals())
        check_output(cmd, shell=True)

    return output_path


def rotate_direction_vec(rotation):
    x, y = 0, 10
    sined = np.sin(rotation)
    cosined = np.cos(rotation)
    normed_x = x*cosined  - y*sined
    normed_y = x*sined    + y*cosined
    return [np.around(normed_x, decimals=2), np.around(normed_y, decimals=2)]


@utils.filepath_cacher
def plot_frame(frame, x, y, rot):
    """

    Args:
        frame (Frame):
        x (list): list of x coordinates to plot
        y (list): list of y coordinates to plot
        rot (list): list of rotations to plot

    Returns:
        path of the plotted frame
    """
    path = extract_single_frame(frame)
    figure = plt.figure()
    plt.imshow(plt.imread(path))
    plt.axis('off')
    rotations = np.array([rotate_direction_vec(rot) for rot in rot])
    plt.quiver(y, x, rotations[:, 1], rotations[:, 0], scale=500, color='yellow')

    video_name = frame.fc.video_name
    uid = uuid.uuid4()
    output_path = f'/tmp/{video_name}-plot-{uid}.png'
    figure.savefig(output_path)
    plt.close()
    return output_path


@utils.filepath_cacher
def plot_video(data):
    """
    Creates a video with information of a track
    Args:
        data (list): Contains a list of dictionaries

    Returns:

    """
    from .models import Frame
    # todo pre-extract all frames
    # todo: command to combine frames kinda sucks, see /tmp/{uid}.mp4
    uid = uuid.uuid4()
    output_folder = f'/tmp/{uid}/'
    os.makedirs(output_folder, exist_ok=True)
    for i, d in enumerate(data):
        frame = Frame.objects.get(frame_id=d['frame_id'])
        path = plot_frame(frame, d['x'], d['y'], d['rot'])

        output_path = os.path.join(output_folder, f'{i:04}.png')
        shutil.copy(path, output_path)

    input_path = os.path.join(output_folder, '%04d.png')
    video_output_path = f'/tmp/{uid}.mp4'
    cmd = config.ffmpeg_frames_to_video.format(input_path=input_path, output_path=video_output_path)
    output = check_output(cmd, shell=True)
    print('Output:', output)

    return video_output_path

