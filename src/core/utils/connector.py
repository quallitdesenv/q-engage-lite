import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
from PIL import Image
from cv2 import VideoCapture, COLOR_BGR2RGB, cvtColor
from enum import Enum


class Connector:
    class StreamType(Enum):
        FFMPEG = 1
        GSTREAMER = 2
    
    def __init__(self, type: 'Connector.StreamType' = None):
        if type is None:
            type = Connector.StreamType.FFMPEG
        self.type = type
        self.connection = None

    def connect(self, rtsp_url: str):
        RTSP_URL = rtsp_url
        if self.type == Connector.StreamType.GSTREAMER:
            pipeline_str = f"""
                rtspsrc location={RTSP_URL} latency=0 !
                rtph264depay ! h264parse ! avdec_h264 !
                videoconvert ! video/x-raw,format=RGB !
                appsink name=sink emit-signals=true max-buffers=1 drop=true
                """
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name("sink")
            pipeline.set_state(Gst.State.PLAYING)
            self.connection = appsink
        elif self.type == Connector.StreamType.FFMPEG:
            self.connection = VideoCapture(RTSP_URL)

    def isOpened(self) -> bool:
        if self.type == Connector.StreamType.GSTREAMER:
            return self.connection is not None
        elif self.type == Connector.StreamType.FFMPEG:
            return self.connection.isOpened()
        return False

    def read(self) -> tuple[bool, Image.Image]:
        if self.type == Connector.StreamType.GSTREAMER:
            sample = self.connection.emit("pull-sample")
            buf = sample.get_buffer()
            caps = sample.get_caps()
            arr = buf.extract_dup(0, buf.get_size())
            width = caps.get_structure(0).get_value('width')
            height = caps.get_structure(0).get_value('height')
            frame = np.frombuffer(arr, np.uint8).reshape((height, width, 3))
            frame = Image.fromarray(frame)
            return True, frame
        elif self.type == Connector.StreamType.FFMPEG:
            ret, frame = self.connection.read()
            if ret:
                frame = cvtColor(frame, COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            return ret, frame
        return False, None

    def from_matrix(self, frame) -> Image:
        return Image.fromarray(frame)

    def to_matrix(self, image: Image) -> any:
        return np.array(image)

    def release(self):
        if self.type == Connector.StreamType.GSTREAMER:
            self.connection.get_parent().set_state(Gst.State.NULL)
        elif self.type == Connector.StreamType.FFMPEG:
            self.connection.release()