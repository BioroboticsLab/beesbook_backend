import os
import uuid
from subprocess import check_output
import matplotlib
import shutil
from multiprocessing import Pool

matplotlib.use('Agg')  # need to be executed before pyplot import, deactivates showing of plot in ipython
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import config


def pool():
    if not hasattr(pool, 'p'):
        pool.p = Pool(config.n_threads)
    return pool.p


def scale(*args):
    args = list(args)
    scaling_constant = config.scale
    for i in range(len(args)):
        args[i] = np.array([int(int(xe) * scaling_constant) for xe in args[i]])
    return args[0] if len(args) == 1 else args


def adjust_cropping_window(xs, ys, keepaspect=True):
    xs, ys = scale(xs, ys)
    pad = config.padding
    left, top, right, bottom = ys.min()-pad, xs.min()-pad, ys.max()+pad, xs.max()+pad

    if keepaspect:
        aspect = config.width / config.height
        w, h = right - left, bottom - top
        diff = w - h * aspect
        if diff == 0:
            pass
        if diff < 0:
            left, right = left - abs(diff)//2, right + abs(diff)//2
            if min(config.width - right, left) < 0:
                diff = abs(left) if left < 0 else config.width - right
                left, right = left + diff, right + diff
        elif diff > 0:
            diff = abs(diff) / aspect
            top, bottom = top - diff // 2, bottom + diff // 2
            if min(config.height - bottom, top) < 0:
                diff = abs(top) if top < 0 else config.height - bottom
                top, bottom = top + diff, bottom + diff

    left, top, right, bottom = [x + x % 2 for x in (left, top, right, bottom)]  # make numbers even for ffmpeg
    left, top, right, bottom = max(left, 0), max(top, 0), min(right, config.width), min(bottom, config.height)
    return left, top, right, bottom


def extract_single_frame(frame):
    """
    Extracts the image to a `Frame`-object.
    Args:
        frame (Frame): The frame which should be extracted.

    Returns: The path to the image.

    """
    video_name = frame.fc.video_name
    video_path = frame.fc.video_path

    os.makedirs(f'/tmp/{video_name}/', exist_ok=True)
    output_path = f'/tmp/{video_name}/{frame.index:04}.jpg'

    if not os.path.exists(output_path):
        cmd = config.ffmpeg_extract_single_frame.format(
            video_path=video_path,
            frame_index=frame.index,
            output_path=output_path
        )
        print('executing: ', cmd)
        output = check_output(cmd, shell=True)
        print('output:', output)

    return output_path


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
        indices = {'{:04}'.format(x) for x in framecontainer.frame_set.values_list('index', flat=True)}
        files = set([os.path.splitext(x)[0] for x in os.listdir(output_path)])

        if indices.issubset(files):
            return output_path

    os.makedirs(output_path, exist_ok=True)
    cmd = config.ffmpeg_extract_all_frames.format(video_path=video_path, output_path=output_path)
    print('executing: ', cmd)
    output = check_output(cmd, shell=True)
    print('output:', output)

    return output_path


def extract_video(frames):
    """
    Extracts a number of frames and makes a video.
    Args:
        frames (list:Frame): list of frames

    Returns:

    """
    uid = uuid.uuid4()
    output_folder = f'/tmp/{uid}'
    os.makedirs(output_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        image_path = frame.get_image_path(extract='all')
        output_path = os.path.join(output_folder, f'{i:04}.jpg')
        shutil.copy(image_path, output_path)

    output_video_path = f'/tmp/{uid}.mp4'
    cmd = config.ffmpeg_frames_to_video.format(
        input_path=f'{output_folder}/%04d.jpg',
        output_path=output_video_path
    )
    print('executing: ', cmd)
    check_output(cmd, shell=True)
    shutil.rmtree(output_folder)

    return output_video_path


def rotate_direction_vec(rotation):
    x, y = 0, 10
    sined = np.sin(rotation)
    cosined = np.cos(rotation)
    normed_x = x*cosined - y*sined
    normed_y = x*sined + y*cosined
    return np.around(normed_x, decimals=2), np.around(normed_y, decimals=2)


def plot_frame(path, x, y, rot, crop_coordinates=None):
    """

    Args:
        path: the image input path
        x (list): list of x coordinates to plot
        y (list): list of y coordinates to plot
        rot (list): list of rotations to plot
        crop_coordinates (tuple): values to crop by. By order: left, top, right, bottom

    Returns:
        path of the plotted frame
    """
    uid = uuid.uuid4()
    output_path = f'/tmp/plot-{uid}.jpg'

    if x is None or y is None:
        shutil.copy(path, output_path)
        return output_path

    x, y = scale(x, y)
    fig, ax = plt.subplots()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.width/dpi, config.height/dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # removes white margin
    ax.imshow(plt.imread(path))
    rotations = np.array([rotate_direction_vec(rot) for rot in rot])
    ax.axis('off')
    ax.quiver(y, x, rotations[:, 1], rotations[:, 0], scale=0.45, color='yellow', units='xy', alpha=0.5)

    fig.savefig(output_path, dpi=dpi)
    plt.close()

    if crop_coordinates is not None:
        args = crop_coordinates
        im = Image.open(output_path)
        im = im.crop(args)
        im.save(output_path)

    return output_path


def plot_video(data, crop=False):
    """
    Creates a video with information of a track
    Args:
        data (list): Contains a list of dictionaries
        crop (bool): Crops the video to the area in which all things happen

    Returns:

    """
    from .models import Frame
    uid = uuid.uuid4()
    output_folder = f'/tmp/{uid}/'
    os.makedirs(output_folder, exist_ok=True)

    crop_coordinates = None
    if crop:
        xs = [x for p in data for x in p.get('x', [])]
        ys = [y for p in data for y in p.get('y', [])]
        crop_coordinates = adjust_cropping_window(xs, ys)

    results = []
    for d in data:
        frame = Frame.objects.get(frame_id=d['frame_id'])
        extract_frames(frame.fc)  # pre extracts all frames out of this framecontainer
        r = pool().apply_async(
            plot_frame,
            (frame.get_image_path(), d.get('x'), d.get('y'), d.get('rot'), crop_coordinates)
        )
        results.append(r)

    paths = [r.get() for r in results]  # wait for all
    for i, path in enumerate(paths):
        output_path = os.path.join(output_folder, f'{i:04}.jpg')
        shutil.move(path, output_path)

    input_path = os.path.join(output_folder, '%04d.jpg')
    video_output_path = f'/tmp/{uid}.mp4'
    cmd = config.ffmpeg_frames_to_video.format(input_path=input_path, output_path=video_output_path)
    print('executing: ', cmd)
    output = check_output(cmd, shell=True)
    print('Output:', output)

    shutil.rmtree(output_folder)  # deleting all temporary files

    return video_output_path
