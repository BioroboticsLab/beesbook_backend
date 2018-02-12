from django.db import models
from bb_binary import load_frame_container
import os
from . import media, utils


class Video(models.Model):
    video_name = models.TextField(primary_key=True)
    video_path = models.TextField()


class FrameContainer(models.Model):
    id = models.AutoField(primary_key=True)
    fc_id = models.DecimalField(max_digits=32, decimal_places=0, db_index=True)
    fc_path = models.TextField()
    video_name = models.TextField()

    def get_binary(self):
        return load_frame_container(self.fc_path)

    @property
    def video_path(self):
        return Video.objects.get(video_name=self.video_name).video_path


class Frame(models.Model):
    frame_id = models.DecimalField(max_digits=32, decimal_places=0, primary_key=True)
    fc = models.ForeignKey(FrameContainer)
    index = models.IntegerField()
    timestamp = models.FloatField()

    def __str__(self):
        return f'{self.frame_id} - {self.index}'

    def get_image(self, scale=1.0, extract='single', format="jpg", extract_n_frames=None):
        """
        Retrieves the image, extracting it if necessary.
        Options for extract are 'single' or 'all'.
        'single' extracts just that frame and returns the path.
        'all' extracts all frames of the containing framecontainer and returns the single frame.
        'n' extracts /extract_n_frames/ frames of the containing framecontainer and returns a single frame.

        Returns:
            utils.ReusableBytesIO object containing the image.
        """
        if not extract in ('single', 'all', 'n'):
            raise ValueError("extract must be in {'all', 'single', 'n')")

        if extract == 'single':
            return media.extract_single_frame(self, scale, format=format)

        if extract == 'all' or extract == 'n':
            begin_frame_id, number_of_frames = None, None
            if extract == 'n':
                begin_frame_id, number_of_frames = self.frame_id, extract_n_frames
            all_images = media.extract_frames(framecontainer=self.fc, scale=scale, format=format, return_frame_id=self.frame_id,
                                                begin_frame_id=begin_frame_id, number_of_frames=number_of_frames)
            return all_images[self.frame_id]

    @property
    def cam_id(self):
        videoname = self.fc.video_name
        assert videoname[0:4] == "Cam_"
        cam_id = int(videoname[4])
        assert (cam_id >= 0)
        assert (cam_id <= 3)
        return cam_id

    @staticmethod
    def get_video(frame_ids, scale=0.5):
        frames = [Frame.objects.get(frame_id=frame_id) for frame_id in frame_ids]
        if len(frame_ids) != len(frames):
            raise ValueError('Some or all frame_ids not found.')

        return media.extract_video(frames, scale=scale)

