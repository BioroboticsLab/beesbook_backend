from wsgiref.util import FileWrapper

from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from . import media

# Create your views here.
from plotter.models import Frame


def get_frame(request):
    frame_id = request.GET.get('frame_id')
    if frame_id is None:
        raise HttpResponseBadRequest('Parameter frame_id required')
    frame = Frame.objects.get(frame_id=frame_id)
    video_path = frame.fc.video_path

    path = media.extract_single_frame(video_path, frame.index)
    fw = FileWrapper(open(path, 'rb'))
    return HttpResponse(fw, content_type='image/png')

