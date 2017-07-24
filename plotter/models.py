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

    def get_image(self, scale=1.0, extract='single'):
        """
        Retrieves the image, extracting it if necessary.
        Options for extract are 'single' or 'all'.
        'single' extracts just that frame and returns the path.
        'all' extracts all frames of the containing framecontainer and returns the single frame.

        Returns:
            utils.ReusableBytesIO object containing the image.
        """
        if not extract in ('single', 'all'):
            raise ValueError("extract must be either 'all' or 'single'")

        if extract == 'single':
            return media.extract_single_frame(self, scale)

        if extract == 'all':
            all_images = media.extract_frames(framecontainer=self.fc, scale=scale)
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

