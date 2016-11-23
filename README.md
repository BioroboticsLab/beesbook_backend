# Intro
This repo allows hosting of a simple api to access the frames of a specific `FrameContainer`.

To use it, open up a ssh tunnel:
```bash
ssh -N -L 5000:localhost:5000 [server_address] # in this case the flip server
```

You can access now the api with a video\_name (`FrameContainer.dataSources[0].filename`) and a frame id:
```
http://localhost:5000/get_frame?filename={video_name}&frame_id={frame_id}
```

To use it with a jupyter notebook is even easier:
```python
from plot import plot_frame
plot_frame(frame_container, frame_id)
```

**Notes**
It can take a longer time to extract the frame. My measurements go up to 2min.
