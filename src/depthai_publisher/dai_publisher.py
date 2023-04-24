#!/usr/bin/env python3

import cv2

import rospy
from sensor_msgs.msg import CompressedImage, Image

from cv_bridge import CvBridge, CvBridgeError
import depthai as dai

import numpy as np


class DepthaiCamera():
    res = [416, 416]
    fps = 20.0

    pub_topic = '/depthai_node/image/compressed'
    pub_topic_raw = '/depthai_node/image/raw'

    def __init__(self):
        self.pipeline = dai.Pipeline()

        # Pulbish ros image data
        self.pub_image = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=10)
        self.pub_image_raw = rospy.Publisher(self.pub_topic_raw, Image, queue_size=10)

        rospy.loginfo("Publishing images to rostopic: {}".format(self.pub_topic))

        self.br = CvBridge()

        rospy.on_shutdown(lambda: self.shutdown())

    def rgb_camera(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(self.res[0], self.res[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(self.fps)

        # Def xout / xin
        ctrl_in = self.pipeline.createXLinkIn()
        ctrl_in.setStreamName("cam_ctrl")
        ctrl_in.out.link(cam_rgb.inputControl)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("video")

        cam_rgb.preview.link(xout_rgb.input)

    def run(self):
        self.rgb_camera()

        with dai.Device(self.pipeline) as device:
            video = device.getOutputQueue(
                name="video", maxSize=1, blocking=False)

            while True:
                frame = video.get().getCvFrame()

                self.publish_to_ros(frame)

    def publish_to_ros(self, frame):
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
        self.pub_image.publish(msg_out)
        # Publish image raw
        msg_img_raw = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
        self.pub_image_raw.publish(msg_img_raw)
        

    def shutdown(self):
        cv2.destroyAllWindows()


def main():
    rospy.init_node('depthai_node')
    dai_cam = DepthaiCamera()

    while not rospy.is_shutdown():
        dai_cam.run()

    dai_cam.shutdown()
