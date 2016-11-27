import os
import requests

import matplotlib.pyplot as plt
from IPython.display import HTML

import config as config


def download_file(url, dl_path):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        return r.status_code

    with open(dl_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

    return r.status_code


def prepare():
    if requests.get(config.API_IS_UP).status_code != 200:
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
        status_code = download_file(u, img_path)
        if status_code != 200:
            raise Exception('Error on serverside')

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
        status_code = download_file(u, video_path)
        if status_code != 200:
            raise Exception('Error on serverside')


    return HTML('''
    <video width="320" height="240" controls>
        <source src="{}" type="video/mp4">
    </video>
    '''.format(video_path))
