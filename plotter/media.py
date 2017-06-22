import os
import tempfile
from subprocess import check_output
import matplotlib
import shutil
import copy
from multiprocessing import Pool

matplotlib.use('Agg')  # need to be executed before pyplot import, deactivates showing of plot in ipython
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
from PIL import Image

from . import config
from . import utils

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


@utils.buffer_object_cacher(key=lambda x: x.frame_id, maxsize=16)
def extract_single_frame(frame):
    """
    Extracts the image belonging to a `Frame`-object.
    Args:
        frame (Frame): The frame which should be extracted.

    Returns:
        An utils.ReusableBytesIO object containing the image.

    """
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmpfile:

        cmd = config.ffmpeg_extract_single_frame.format(
            video_path=frame.fc.video_path,
            frame_index=frame.index,
            output_path=tmpfile.name
        )
        print('executing: ', cmd)
        output = check_output(cmd, shell=True)
        print('output:', output)

        with open(tmpfile.name, "rb") as file:
            buf = utils.ReusableBytesIO(file.read())
            buf.seek(0)
            return buf

@utils.buffer_object_cacher(key=lambda x: x.video_path, maxsize=32)
def extract_frames(framecontainer):
    """
    Extracts all frame-images of the corresponding video file of a FrameContainer.

    Args:
        framecontainer (FrameContainer): The FrameContainer which represents the video file from which the frames
         should be extracted

    Returns:
        Dictionary with a mapping of Frame.id to utils.ReusableBytesIO object containing the frame.

    """
    video_name = framecontainer.video_name

    # Required frames.
    # Subset of the resulting filenames of the ffmpeg command.
    frame_set = framecontainer.frame_set
    images = ['{:04}.jpg'.format(x) for x in frame_set.values_list('index', flat=True)]
    
    results = dict()

    with tempfile.TemporaryDirectory() as tmpdir:

        cmd = config.ffmpeg_extract_all_frames.format(
            video_path=framecontainer.video_path, output_path=tmpdir)
        print('executing: ', cmd)
        output = check_output(cmd, shell=True)
        print('output:', output)
        
        for idx, frame in enumerate(frame_set.all()):
            with open(os.path.join(tmpdir, images[idx]), "rb") as file:
                output = utils.ReusableBytesIO(file.read())
                output.seek(0)
                results[frame.frame_id] = output
    return results


def extract_video(frames):
    """
    Extracts a number of frames and makes a video.
    Args:
        frames (list:Frame): list of frames

    Returns:
        The video as a utils.ReusableBytesIO object.
    """
    with tempfile.TemporaryDirectory() as tmpdir:

        for i, frame in enumerate(frames):
            buffer = frame.get_image(extract='all')
            output_path = os.path.join(tmpdir, f'{i:04}.jpg')
            shutil.copyfileobj(buffer, output_path)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpfile:

            cmd = config.ffmpeg_frames_to_video.format(
                input_path=f'{tmpdir}/%04d.jpg',
                output_path=tmpfile.name
            )
            print('executing: ', cmd)
            check_output(cmd, shell=True)
            
            with open(tmpfile.name, "rb") as file:
                output = utils.ReusableBytesIO(file.read())
                output.seek(0)
                return output

def rotate_direction_vec(rotation):
    x, y = 0, 10
    sined = np.sin(rotation)
    cosined = np.cos(rotation)
    normed_x = x*cosined - y*sined
    normed_y = x*sined + y*cosined
    return np.around(normed_x, decimals=2), np.around(normed_y, decimals=2)


def plot_frame(buffer, x=None, y=None, rot=None, crop_coordinates=None, color=None, radius=None, label=None, title=None, **args):
    """

    Args:
        buffer: utils.ReusableBytesIO object containing the image
        x (list): list of x coordinates to plot
        y (list): list of y coordinates to plot
        rot (list): list of rotations to plot
        crop_coordinates (tuple): values to crop by. By order: left, top, right, bottom

    Returns:
        utils.ReusableBytesIO object containing the final image
    """
    
    plot_bees = False
    if x and y:
        x, y = scale(x, y)
        plot_bees = True
    
    outputbuffer = None

    if plot_bees or (title is not None):
        fig, ax = plt.subplots()
        dpi = fig.get_dpi()
        fig.set_size_inches(config.width/dpi, config.height/dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # removes white margin
        image = plt.imread(buffer, format="JPG")
        ax.imshow(image)
        
        ax.axis('off')
        if plot_bees:
            # Draw arrows if rotation is given.
            if rot is not None:
                rotations = np.array([rotate_direction_vec(rot) for rot in rot])
                ax.quiver(y, x, rotations[:, 1], rotations[:, 0], scale=0.45, color='yellow', units='xy', alpha=0.5)
            # Draw scatterplot if radius is given.
            if color is None:
                color = ["yellow"] * x.shape[0]
            color = np.array(color)
            for unique_color in np.unique(color):
                idx = color == unique_color
                if radius is not None:
                    radius = np.array(radius)
                    ax.scatter(y[idx], x[idx], facecolors='none', edgecolors=unique_color, marker="o", s=radius[idx][0])
                
                if label is not None:
                    label = np.array(label)
                    label = np.array(label)
                    for i, label_i in enumerate(label[idx]):
                        if label_i is None or not label_i:
                            continue
                        ax.text(y[idx][i], x[idx][i], label_i, color=unique_color, fontsize=36, alpha=0.5)
        if title is not None:
            txt = plt.text(0.1, 0.9, title, size=36, color='white', transform=ax.transAxes, horizontalalignment='left')
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground='k')])
        # Make sure that the plot is cropped at the image's bounds.
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((0, image.shape[0]))
        
        outputbuffer = utils.ReusableBytesIO()
        fig.savefig(outputbuffer, dpi=dpi, format='JPG')
        plt.close()
    else:
        outputbuffer = copy.copy(buffer)

    outputbuffer.seek(0)

    if crop_coordinates is not None:
        im = Image.open(outputbuffer)
        im = im.crop(crop_coordinates)
        
        outputbuffer.seek(0)
        with open(outputbuffer, "wb") as file:
            im.save(file, format='JPG')
        outputbuffer.seek(0)

    return outputbuffer


def plot_video(data, crop=False):
    """
    Creates a video with information of a track
    Args:
        data (list): Contains a list of dictionaries
        crop (bool): Crops the video to the area in which all things happen

    Returns:
        utils.ReusableBytesIO object containing the video.
    """
    from .models import Frame

    crop_coordinates = None
    if crop:
        xs = [x for p in data for x in p.get('x', [])]
        ys = [y for p in data for y in p.get('y', [])]
        crop_coordinates = adjust_cropping_window(xs, ys)

    results = []
    extracted_frames = dict()
    for d in data:
        d["crop_coordinates"] = crop_coordinates
        frame = Frame.objects.get(frame_id=d['frame_id'])

        if frame.frame_id not in extracted_frames:
            extracted_frames = {**extracted_frames, **extract_frames(frame.fc)}
            assert(frame.frame_id in extracted_frames)

        r = pool().apply_async(
            plot_frame,
            (extracted_frames[frame.frame_id],),
            d
            #d.get('x'), d.get('y'), d.get('rot'), crop_coordinates)
        )
        results.append(r)

    images = [r.get() for r in results]  # wait for all

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write buffer to disk for ffmpeg to work.
        for idx, buffer in enumerate(images):
            with open(os.path.join(tmpdir, f'{idx:04d}.jpg'), "wb") as file:
                shutil.copyfileobj(buffer, file)
        
        input_path = os.path.join(tmpdir, '%04d.jpg')
        video_output_path = os.path.join(tmpdir, 'video.mp4')
        cmd = config.ffmpeg_frames_to_video.format(input_path=input_path, output_path=video_output_path)
        print('executing: ', cmd)
        output = check_output(cmd, shell=True)
        print('Output:', output)

        with open(video_output_path, "rb") as file:
            buf = utils.ReusableBytesIO(file.read())
            buf.seek(0)
            return buf
