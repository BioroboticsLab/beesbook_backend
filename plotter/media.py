import os
import tempfile
from subprocess import check_output
import matplotlib
import shutil
import copy
from multiprocessing import Pool
import math

matplotlib.use('Agg')  # need to be executed before pyplot import, deactivates showing of plot in ipython
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
from PIL import Image

from . import config
from . import utils
from . import api

def pool():
    if not hasattr(pool, 'p'):
        pool.p = Pool(config.n_threads)
    return pool.p

def adjust_cropping_window(xs, ys, scale, keepaspect=True, padding=600):
    xs, ys = (xs * scale).astype(np.int), (ys * scale).astype(np.int)
    padding *= scale

    width, height = int(config.width * scale), int(config.height * scale)

    left, top, right, bottom = ys.min()-padding, xs.min()-padding,\
                               ys.max()+padding, xs.max()+padding
    
    if keepaspect:
        aspect = width / height
        w, h = right - left, bottom - top
        diff = w - h * aspect
        if diff == 0:
            pass
        if diff < 0:
            left, right = left - abs(diff)//2, right + abs(diff)//2
            if min(width - right, left) < 0:
                diff = abs(left) if left < 0 else width - right
                left, right = left + diff, right + diff
        elif diff > 0:
            diff = abs(diff) / aspect
            top, bottom = top - diff // 2, bottom + diff // 2
            if min(height - bottom, top) < 0:
                diff = abs(top) if top < 0 else height - bottom
                top, bottom = top + diff, bottom + diff

    left, top, right, bottom = [x + x % 2 for x in (left, top, right, bottom)]  # make numbers even for ffmpeg
    left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
    return left, top, right, bottom


@utils.buffer_object_cacher(key=lambda frame, scale: (frame.frame_id, scale), maxsize=16)
def extract_single_frame(frame, scale):
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
            output_path=tmpfile.name,
            scale=scale
        )
        print('executing: ', cmd)
        output = check_output(cmd, shell=True)
        print('output:', output)

        with open(tmpfile.name, "rb") as file:
            buf = utils.ReusableBytesIO(file.read())
            buf.seek(0)
            return buf

@utils.buffer_object_cacher(key=lambda framecontainer, scale: (framecontainer.video_path, scale), maxsize=32)
def extract_frames(framecontainer, scale):
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
            video_path=framecontainer.video_path, output_path=tmpdir, scale=scale)
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
            with open(output_path, "wb") as file:
                shutil.copyfileobj(buffer, file)

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

def rotate_direction_vec(rotation, scale):
    x, y = 0, 5 / scale
    sined = np.sin(rotation)
    cosined = np.cos(rotation)
    normed_x = x*cosined - y*sined
    normed_y = x*sined + y*cosined
    return np.around(normed_x, decimals=2), np.around(normed_y, decimals=2)


class FramePlotter(api.FramePlotter):

    # Internal attributes.
    _xs_scaled = None
    _ys_scaled = None

    def __init__(self, **args):
        super(FramePlotter, self).__init__(**args)

        if self._colors is not None:
            self._colors = np.array(self._colors)
        if self._labels is not None:
            self._labels = np.array(self._labels)

    # Wrap the internal properties in case they require post-processing.

    @property
    def xs(self):
        if not self._xs:
            return None
        if self._xs_scaled is None:
            self._xs_scaled = (np.array(self._xs) * self._scale).astype(np.int)
        return self._xs_scaled
    @property
    def ys(self):
        if not self._ys:
            return None
        if self._ys_scaled is None:
            self._ys_scaled = (np.array(self._ys) * self._scale).astype(np.int)
        return self._ys_scaled
    @property
    def angles(self):
        return self._angles
    @property
    def sizes(self):
        if self._sizes is None:
            return None
            self._sizes = np.array(self._sizes) / self.scale
        return self._sizes
    @property
    def colors(self):
        if self._colors is None:
            self._colors = np.array(["yellow"] * self.xs.shape[0])
        return self._colors
    @property
    def labels(self):
        return self._labels
    @property
    def title(self):
        return self._title
    @property
    def frame_id(self):
        return self._frame_id
    @property
    def scale(self):
        return self._scale
    @property
    def crop_coordinates(self):
        if self._crop_coordinates is None:
            return self._crop_coordinates
        return list(np.array(self._crop_coordinates) * self.scale)
    @property
    def width(self):
        return int(config.width * self.scale)
    @property
    def height(self):
        return int(config.height* self.scale)

    def plot(self, buffer):
        """

        Args:
            buffer: file-like object containing the image

        Returns:
            utils.ReusableBytesIO object containing the final image
        """
       
        outputbuffer = None

        fig, ax = plt.subplots()
        dpi = fig.get_dpi()
        fig.set_size_inches(self.width/dpi, self.height/dpi)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # removes white margin
        image = plt.imread(buffer, format="JPG")
        ax.imshow(image)
        ax.axis('off')

        if self.xs is not None and self.ys is not None:
            width = image.shape[1]
            height = image.shape[0]

            if self.crop_coordinates:
                x, y, x2, y2 = self.crop_coordinates
                width = x2 - x
                height = y2 - y
            # Draw arrows if rotation is given.
            if self.angles is not None:
                rotations = np.array([rotate_direction_vec(rot, self.scale) for rot in self.angles])
                ax.quiver(self.ys, self.xs, rotations[:, 1], rotations[:, 0], scale=0.45 / self.scale, color=self.colors, units='xy', alpha=0.5)
            
            for unique_color in np.unique(self.colors):
                idx = self.colors == unique_color

                # Draw scatterplot if radius is given.
                if self.sizes is not None:
                    radius = np.array(self.sizes)
                    # The size is meant to be in pixels of the original video.
                    # A radius of around 40 pixels would be a tag.
                    size = float(radius[idx][0])
                    # Adjust for cropping region.
                    # Usually the markersize scales with the window.
                    size /= width / self.scale / config.width
                    # Calcluate area, adjusted for scaling factor.
                    size = (size * self.scale) ** 2.0
                    ax.scatter(self.ys[idx], self.xs[idx], facecolors='none', edgecolors=unique_color, marker="o",
                      s=size, linewidth=10 * self.scale)
                # Draw marker labels if given.
                if self.labels is not None:
                      for i, label_i in enumerate(self.labels[idx]):
                        if label_i is None or not label_i:
                            continue
                        ax.text(self.ys[idx][i], self.xs[idx][i], label_i, color=unique_color, fontsize=int(72 * self.scale), alpha=0.5)
        if self.title is not None:
            txt = plt.text(0.1, 0.9, self.title, size=int(108 * self.scale), color='white', transform=ax.transAxes, horizontalalignment='left')
            txt.set_path_effects([matplotlib.patheffects.withStroke(linewidth=5, foreground='k')])
        if self.crop_coordinates is not None:
            x, y, x2, y2 = self.crop_coordinates
            ax.set_xlim((x, x2))
            ax.set_ylim((y, y2))
        else:
            # Make sure that the plot is cropped at the image's bounds.
            ax.set_xlim((0, image.shape[1]))
            ax.set_ylim((0, image.shape[0]))
        
        outputbuffer = utils.ReusableBytesIO()
        fig.savefig(outputbuffer, dpi=dpi, format='JPG')
        plt.close()

        outputbuffer.seek(0)

        if False and self.crop_coordinates is not None:
            im = Image.open(outputbuffer)
            im = im.crop(self.crop_coordinates)
        
            outputbuffer.seek(0)
            im.save(outputbuffer, format='JPEG')
            outputbuffer.seek(0)

        return outputbuffer


class VideoPlotter(api.VideoPlotter):
    
    def __init__(self, **args):
        super(VideoPlotter, self).__init__(**args)

        # 'frames' can be a list of dictionaries, too.
        if len(self._frames) > 0 and isinstance(self._frames[0], dict):
            self._frames = [FramePlotter.from_dict(frame) for frame in self._frames]

        # First, fill in missing frames if requested.
        if self._fill_gaps:
            from .models import Frame

            fids = [frame.frame_id for frame in self._frames]
            i = 0
            while i < len(fids) - 1:
                fid1, fid2 = fids[i], fids[i+1]
                f1 = Frame.objects.get(frame_id=fid1)
                f2 = Frame.objects.get(frame_id=fid2)
                if f1.fc_id != f2.fc_id:
                    i += 1
                    continue
                if f2.index - f1.index == 1:
                    i += 1
                    continue
                fill_frame_ids = (
                    Frame.objects.filter(
                        fc_id=f1.fc_id,
                        index__gt=f1.index,
                        index__lt=f2.index
                    ).order_by('index').values_list('frame_id', flat=True)
                )
                for fill_frame_id in reversed(fill_frame_ids):  # reversed so we dont need to increment i
                    fill_frame_id = int(fill_frame_id)
                    fids.insert(i+1, fill_frame_id)
                    # Fill data with copy of previous frame.
                    filler_frame = copy.deepcopy(self._frames[i])
                    filler_frames._frame_id = fill_frame_id
                    self._frames.insert(i+1, filler_frames)
                i += 1 + len(fill_frame_ids)

        # Calculate auto-cropping.
        if self._crop_margin is not None:
            scale = self._scale
            if scale is None and len(self._frames) > 0:
                scale = self._frames[0]._scale
            xs = np.array([x for frame in self._frames for x in frame._xs])
            ys = np.array([y for frame in self._frames for y in frame._ys])
            self._crop_coordinates = adjust_cropping_window(xs, ys,
                                        scale=scale, padding=self._crop_margin)

        # Some options can be set for all frames through the video options.
        for property in ("_crop_coordinates", "_scale"):
            value = getattr(self, property)
            if value is not None:
                for frame in self._frames:
                    if getattr(frame, property) is None:
                        setattr(frame, property, value)
        
    def plot(self):
        """
        Creates a video with information of a track

        Returns:
            utils.ReusableBytesIO object containing the video.
        """
        from .models import Frame

        results = []
        extracted_frames = dict()
        for plotter in self._frames:
            frame = Frame.objects.get(frame_id=plotter.frame_id)

            if frame.frame_id not in extracted_frames:
                extracted_frames = {**extracted_frames, **extract_frames(frame.fc, plotter.scale)}
                assert(frame.frame_id in extracted_frames)

            r = pool().apply_async(
                plotter.plot,
                (extracted_frames[frame.frame_id],)
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
