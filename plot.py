import os
import urllib

import matplotlib.pyplot as plt

import config as config


def plot_frame(frame_container, frame_id):
    if urllib.urlopen(config.API_IS_UP).getcode() != 200:
        raise Exception('Cannot access endpoint. SSH Tunnel probably not running.\n'
                        '(ssh -N -L 5000:localhost:5000 [server_address])')

    if not os.path.exists(config.FRAMES_DIR):
        os.makedirs(config.FRAMES_DIR)

    video_name = frame_container.dataSources[0].filename
    name, ext = os.path.splitext(video_name)
    img_path = '{}/{}-frame_{}.{}'.format(config.FRAMES_DIR, name, frame_id, 'png')

    if not os.path.exists(img_path):
        u = config.ENDPOINT.format(video_name=video_name, frame_id=frame_id)
        urllib.urlretrieve(u, img_path)

    return plt.imshow(plt.imread(img_path))