import uuid
from subprocess import check_output
import os
import matplotlib
matplotlib.use('Agg') # need to be executed before pyplot import, deactivates showing of plot in ipython
import matplotlib.pyplot as plt

import numpy as np



from plotter import utils

GPU = False
if GPU:
    from . import config_gpu as config
else:
    from . import config


def extract_frame(video_path, frame_index):
    name = utils.get_filename(video_path)

    output_path = f'/tmp/{name}_{frame_index}.png'
    cmd = config.ffmpeg_frame_cmd.format(**locals())

    if not os.path.exists(output_path):
        output = check_output(cmd, shell=True)
        print('output:', output)

    return output_path

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
def plot_frame(video_path, frame_index, x: list, y:list, rot:list):
    path = extract_frame(video_path, frame_index)
    figure = plt.figure()
    plt.imshow(plt.imread(path))
    plt.axis('off')
    rotations = np.array([rotate_direction_vec(rot) for rot in rot])
    plt.quiver(y, x, rotations[:, 1], rotations[:, 0], scale=500, color='yellow')

    name = utils.get_filename(video_path)
    id = str(uuid.uuid4())
    output_path = f'/tmp/{name}-plot-{id}.png'
    figure.savefig(output_path)
    return output_path