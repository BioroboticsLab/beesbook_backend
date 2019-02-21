import json
import io

server_adress = '127.0.0.1:8000'

def get_image_origin(cam_id, year=2016):
    """Returns the origin of the recorded images.
        (0, 0) is top-left corner and (1, 1) bottom-right.
        Origins taken from https://git.imp.fu-berlin.de/bioroboticslab/organization/wikis/Experiment%20pictures"""
    if year == 2016:
        return [(0, 1), (1, 0), (0, 1), (1, 0)][cam_id]
    raise ValueError("Unknown year.")

def get_plot_coordinates(x, y):
    """Transform x, y coordinates to plot in the images' coordinate system."""
    return y, x

def transform_axis_coordinates(cam_id=None, year=2016, ax=None, origin=None):
    """Transforms an axis (or the current axis) for plotting in the images' coordinate system."""
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    origin = origin or get_image_origin(cam_id, year)
    if origin[0] == 1:
        ax.invert_xaxis()
    if origin[1] == 0:
        ax.invert_yaxis()

class _ObjectRequester(object):
    def execute_request(self, command, method='POST', data=None):
        import requests

        url = ('http://' + server_adress + '/plotter/{}/').format(command)

        if method == 'GET':
            r = requests.get(url, stream=True)
        elif method == 'POST':
            r = requests.post(url, data=data, stream=True)
        else:
            raise Exception('"{}" method not implemented'.format(method))
        
        if r.status_code != 200:
            raise Exception("HTTP Code: {}".format(r.status_code))

        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                buf.write(chunk)
        buf.seek(0)

        return buf

class FramePlotter(_ObjectRequester):

    # The following attributes are global image attributes.
    _frame_id = None         # bb_binary frame id.
    _title = None            # Text to plot in the upper left corner.
    _scale = None            # Resizing of the image prior to plotting.
    _crop_coordinates = None # Allows displaying only a small part of the image.
    _crop_mode = None        # Default "shift": keep cropping region valid, "crop": crop out smaller image at border.
    _path_alpha = None       # The base transparency of the paths.
    _raw = None              # Requests a near-lossless image without plotting.
    _no_rotate = None        # The image won't be rotated to hive coordinates (use with raw=True).
    _decode_all_frames = None # Whether to decode and cache the complete video. 
    _decode_n_frames = None  # Whether to decode and cache the subsequent n frames.

    # The following attributes are vectors.
    _xs, _ys = None, None    # Positions of markers.
    _angles = None           # For arrows: rotation of markers.
    _sizes = None            # For circles: radius of circle.
    _colors = None           # matplotlib colors.
    _labels = None           # Text to print at each marker.

    # Internal attributes.
    _paths = None            # Generated by VideoPlotter.track_labels.

    def __init__(self, **args):
        """
        Arguments:
            frame_id: Database/bb_binary frame id of the image.
            title: Text that will be displayed in the image (or "auto" for automatic title).
            scale: Factor to resize the returned image (default = 0.5).
            crop_coordinates: Image pixel coordinates of the region to crop.
            crop_mode: "shift": Cropping regions that are out of the image bounds will be shifted (default),
                        "crop": Crops at the image border will yield smaller subimages.
            path_alpha: Opacity of the traced tracks (default = 0.25).
            raw: Whether to return the raw image data as a numpy array without compressing it to JPEG.
            no_rotate: Only with raw=True. Whether to return the image in the original (camera) orientation.
            decode_all_frames: Whether to proactively decode and cache all frames of the video.
            decode_n_frames: Whether to decode and cache the next n frames of the video.
            xs: List of x coordinates (in image pixels).
            ys: List of y coodinates (in image pixels).
            angles: List of orientations of arrows to be drawn. Same order as xs/ys.
            sizes: List of sizes (radius in pixels) of small circles for every point in xs/ys.
            colors: List of matplotlib colors for the markers of the points xs/ys.
            labels: Text to draw above the markers in xs/ys.
        """
        for property in ("xs", "ys",
                         "angles", "sizes", "colors", "labels",
                         "frame_id", "title", "scale", "crop_coordinates", "crop_mode",
                         "path_alpha", "raw", "no_rotate", "decode_all_frames", "decode_n_frames"):
            if property not in args:
                continue
            setattr(self, "_" + property, args[property])
        
        if self._no_rotate and not self._raw:
            raise ValueError("no_rotate=True requires raw=True.")
        if self._crop_mode:
            if self._crop_mode not in ("shift", "crop"):
                raise ValueError("crop_mode must be one of 'shift', 'crop'.")
        if self._frame_id is not None:
            self._frame_id = int(self._frame_id)
            
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    @classmethod
    def from_json(cls, data_json):
        data = json.loads(data_json)
        return cls.from_dict(data)
    def to_json(self):
        return json.dumps(dict(self))

    # Retrieve all properties as (name, value) pairs.
    # Used to convert to dictionary.
    def __iter__(self):
        def all_attributes():
            yield "frame_id", self._frame_id
            yield "xs", self._xs
            yield "ys", self._ys
            yield "angles", self._angles
            yield "sizes", self._sizes
            yield "colors", self._colors
            yield "labels", self._labels
            yield "title", self._title
            yield "scale", self._scale
            yield "crop_coordinates", self._crop_coordinates
            yield "crop_mode", self._crop_mode
            yield "path_alpha", self._path_alpha
            yield "raw", self._raw
            yield "no_rotate", self._no_rotate
            yield "decode_all_frames", self._decode_all_frames
            yield "decode_n_frames", self._decode_n_frames

        for (name, value) in all_attributes():
            if value is not None:
                yield (name, value)

    def get_image(self):
        """
            Requests the image from the backend server.
                
            Returns:
                numpy array containing the image
        """
        data = dict(frame_options=self.to_json())
        buf = self.execute_request("plot_frame", data=data)
        if not self._raw:
            import matplotlib.pyplot as plt
            return plt.imread(buf, format="JPG")
        else:
            import numpy as np
            return np.load(buf, allow_pickle=False)

class VideoPlotter(_ObjectRequester):

    # List of FramePlotter containing all required options.
    _frames = None
    # Auto-crop margin around the supplied coordinates.
    _crop_margin = None
    # Whether to automatically fill in missing frames.
    _fill_gaps = True
    # Margin around the specified frames.
    # This allows to e.g. get a video /around/ a supplied frame.
    _n_frames_before_after = None
    # Whether to draw tracks defined by the labels.
    _track_labels = None
    # Prefix that is added to all frame titles.
    # Can be 'auto' for an automated title containing frame index, date, time and camera ID.
    _title = None
    # The framerate of the video. Realtime is 3 and higher is faster.
    _framerate = None

    # The following attributes can overwrite frame options.
    _crop_coordinates = None
    _scale = None
    _path_alpha = None

    def __init__(self, **args):
        """
        Arguments:
            frames: List of FramePlotter objects to concatenate into a video.
            crop_margin: Auto-crop margin in pixels around the positions from the frames.
            fill_gaps: Whether to automatically fetch and insert missing frame ids.
            n_frames_before_after: Whether to automatically fetch the previous and next n frames of the first and last given frame.
            track_labels: Whether to automatically connect the same labels in the FramePlotters with lines.
            title: Text to prepend to all frames' titles (can be "auto" for index, date, time, camera).
            framerate: The framerate of the resulting video (default 3).
            crop_coordinates: Passed to the individual FramePlotters.
            scale: Passed to the individual FramePlotters.
            path_alpha: Passed to the individual FramePlotters.
        """
        for property in ("frames", "crop_margin",
                         "fill_gaps", "track_labels", "crop_coordinates", "scale", "title",
                         "framerate", "path_alpha", "n_frames_before_after"):
            if property not in args:
                continue
            setattr(self, "_" + property, args[property])

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    @classmethod
    def from_json(cls, data_json):
        return cls.from_dict(json.loads(data_json))
    def to_json(self):
        dictionary = dict(self)
        dictionary["frames"] = [dict(f) for f in dictionary["frames"]]
        return json.dumps(dictionary)

    # Retrieve all properties as (name, value) pairs.
    def __iter__(self):
        def all_attributes():
            yield "frames", self._frames
            yield "crop_margin", self._crop_margin
            yield "fill_gaps", self._fill_gaps
            yield "n_frames_before_after", self._n_frames_before_after
            yield "track_labels", self._track_labels
            yield "crop_coordinates", self._crop_coordinates
            yield "scale", self._scale
            yield "title", self._title
            yield "framerate", self._framerate
            yield "path_alpha", self._path_alpha

        for (name, value) in all_attributes():
            if value is not None:
                yield (name, value)

    def get_video(self, display_in_notebook=True, save_to_path='video_plot.mp4', display_scale=0.15):
        """
            Requests the video from the backend server.

            Args:
                display_in_notebook: whether to show the video in a notebook - must be saved to disk in order to do that
                save_to_path: path to save to video to; required if displaying in notebook
                display_scale: scaling of the notebook display

            Returns:
                io.BytesIO object containing the video data.
                This object might be closed if the video was saved to disk.
        """
        data = dict(video_options=self.to_json())
        buf = self.execute_request("plot_video", data=data)
        
        if save_to_path is not None:
            import shutil
            with open(save_to_path, "wb") as file:
                shutil.copyfileobj(buf, file)
            
        if display_in_notebook:
            import random
            from IPython.display import HTML, display
            VIDEO_HTML = """
            <video style='margin: 0 auto;' width="{width}" height="{height}" controls>
                <source src="{src}" type="video/mp4">
            </video>
            """
            display(HTML(VIDEO_HTML.format(
                    src="{}?{}".format(save_to_path, random.randint(0, 99999)),
                    width=int(4000 * display_scale),
                    height=int(3000 * display_scale)
                )))

        return buf
