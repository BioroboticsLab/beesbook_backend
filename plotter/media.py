import os
import uuid
from subprocess import check_output

import matplotlib

from plotter.models import Frame, FrameContainer

matplotlib.use('Agg')  # need to be executed before pyplot import, deactivates showing of plot in ipython
import matplotlib.pyplot as plt
import numpy as np


from plotter import utils

GPU = False
if GPU:
    from . import config_gpu as config
else:
    from . import config


def extract_frames(framecontainer: FrameContainer):
    """
    Extract multiple frames.
    """
    video_name = framecontainer.video_name
    video_path = framecontainer.video_path
    output_path = f'/tmp/{video_name}'

    # check if files already exist
    if len(os.listdir(output_path)) == Frame.objects.filter(fc=framecontainer).count():
        return output_path

    os.makedirs(output_path, exist_ok=True)
    cmd = config.ffmpeg_extract_all_frames.format(video_path=video_path, output_path=output_path)
    output = check_output(cmd, shell=True)
    print('output:', output)

    frame_ids = [frame.frame_id for frame in Frame.objects.filter(fc=framecontainer).order_by('index')]
    images = os.listdir(output_path)
    for image_path, frame_id in zip(images, frame_ids):
        os.replace(image_path, f'{output_path}/{frame_id}.png')

    return output_path


def extract_single_frame(frame: Frame):
    video_name = frame.fc.video_name
    video_path = frame.fc.video_path

    output_path = f'/tmp/{video_name}/{frame.id}.png'
    cmd = config.ffmpeg_extract_single_frame.format(
        video_path=video_path,
        frame_index=frame.index,
        output_path=output_path
    )

    if not os.path.exists(output_path):
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
def plot_frame(frame: Frame, x: list, y: list, rot: list):
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
    return output_path
