import os
import urllib

import matplotlib.pyplot as plt
from IPython.display import HTML

import config as config


def prepare():
    if urllib.urlopen(config.API_IS_UP).getcode() != 200:
        raise Exception('Cannot access endpoint. SSH Tunnel probably not running.\n'
                        '(ssh -N -L 5000:localhost:5000 [server_address])')

    if not os.path.exists(config.FRAMES_DIR):
        os.makedirs(config.FRAMES_DIR)


def plot_frame(frame_container, frame):
    if len(frame_container.frames) <= frame or frame < 0:
        raise ValueError('Parameter frame does not exist in FrameContainer')

    prepare()

    video_name = frame_container.dataSources[0].filename
    name, ext = os.path.splitext(video_name)
    img_path = ('{}/'+config.img_name).format(config.FRAMES_DIR, name, frame)

    if not os.path.exists(img_path):
        u = config.IMAGE_ENDPOINT.format(video_name=video_name, frame=frame)
        urllib.urlretrieve(u, img_path)

    return plt.imshow(plt.imread(img_path))


def plot_video(frame_container, left_frame, right_frame):
    if len(frame_container.frames) <= right_frame or right_frame < 0:
        raise ValueError('Chosen frames are not valid.')

    if right_frame - left_frame < 2:
        raise ValueError('to_frame has to be greater than from_frame by value 2')

    prepare()

    video_name = frame_container.dataSources[0].filename
    name, ext = os.path.splitext(video_name)
    video_path = ('{base_dir}/' + config.video_name).format(
        base_dir=config.FRAMES_DIR,
        name=name,
        left_frame=left_frame,
        right_frame=right_frame
    )

    if not os.path.exists(video_path):
        u = config.VIDEO_ENDPOINT.format(video_name=video_name, left_frame=left_frame, right_frame=right_frame)
        urllib.urlretrieve(u, video_path)

    return HTML('''
    <video width="320" height="240" controls>
        <source src="{}" type="video/mp4">
    </video>
    '''.format(video_path))
