import json
from wsgiref.util import FileWrapper

from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from plotter.models import Frame, FrameContainer


def get_frame(request):
    frame_id = request.GET.get('frame_id')
    if frame_id is None:
        raise HttpResponseBadRequest('Parameter frame_id required')
    frame = Frame.objects.get(frame_id=frame_id)
    video_path = frame.fc.video_path
@csrf_exempt
def get_video(request):
    if request.method != 'POST':
        raise HttpResponseBadRequest('Only POST requests allowed.')

    frame_ids_param = request.POST.get('frame_ids', None)
    frame_container_id = request.POST.get('frame_container_id', None)

    if not (frame_ids_param or frame_container_id):
        raise HttpResponseBadRequest('Either parameter frame_ids_param or frame_container_id required')

    if frame_ids_param:
        if not isinstance(frame_ids_param, str):
            raise HttpResponseBadRequest('Parameter frame_ids_param has to be json-array string containing frame_ids.')
        frame_ids = json.loads(frame_ids_param)
    else:
        if not isinstance(frame_container_id, int):
            raise HttpResponseBadRequest('Parameter frame_container_id has to be an int')
        fc = FrameContainer.objects.get(fc_id=frame_container_id)
        frame_ids = fc.frame_set.all().values_list('frame_id', flat=True)

    path = Frame.get_video_path(frame_ids)
    return HttpResponse(FileWrapper(open(path, 'rb')), content_type='video/mp4')



