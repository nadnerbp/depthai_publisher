#!/usr/bin/env python3
import cv2
import rospy
import numpy as np
import threading
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from collections import defaultdict, deque

class ArucoDetector():
    frame_sub_topic = '/depthai_node/image/compressed'

    def __init__(self):
        rospy.loginfo("Initialising Aruco Detector...")

        dict_id = cv2.aruco.DICT_5X5_100  # dictionary ID (adjust if needed)

        # --- Handle both new and old OpenCV ArUco APIs ---
        try:
            # New API (OpenCV >= 4.7)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self._use_new_api = True
            rospy.loginfo("Using OpenCV >= 4.7 ArUco API")
        except AttributeError:
            # Old API fallback (OpenCV <= 4.6)
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self._use_new_api = False
            rospy.loginfo("Using legacy OpenCV ArUco API")

        # Publishers
        self.aruco_pub_img = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
        self.aruco_pub_detection = rospy.Publisher(
            '/aruco_detection', Float32MultiArray, queue_size=10)

        # Subscriber (frames only; refined-pose callbacks removed)
        self.frame_sub = rospy.Subscriber(
            self.frame_sub_topic, CompressedImage, self.img_callback,
            queue_size=1, buff_size=2**24)

        # CvBridge + threading
        self.br = CvBridge()
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

        # State
        self.published_ids = set()                                # one-shot publish per ID
        self.coordinate_buffers = defaultdict(lambda: deque(maxlen=5))  # 5-frame buffer per ID

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

        rospy.loginfo("Aruco Detector initialised.")

    def img_callback(self, msg_in):
        try:
            frame = self.br.compressed_imgmsg_to_cv2(msg_in)
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion error: {e}")
            return

        with self.lock:
            self.frame = frame
            self.new_frame_event.set()

    def process_frames(self):
        while not rospy.is_shutdown():
            self.new_frame_event.wait()
            with self.lock:
                frame = self.frame.copy()
                self.new_frame_event.clear()

            processed_frame = self.find_aruco(frame)
            self.publish_to_ros(processed_frame)

    def find_aruco(self, frame):
        # Detection depending on OpenCV version
        if self._use_new_api:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict,
                                                      parameters=self.aruco_params)

        if ids is not None and len(corners) > 0:
            ids = ids.flatten()
            for (marker_corner, marker_id) in zip(corners, ids):
                pts = marker_corner.reshape((4, 2)).astype(int)
                (tl, tr, br, bl) = pts

                # Draw marker and ID
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.putText(frame, str(marker_id), (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

                # Save for averaging
                self.coordinate_buffers[marker_id].append(pts.flatten())

                # Publish after 5 frames (one-shot per ID)
                if (len(self.coordinate_buffers[marker_id]) == 5
                        and marker_id not in self.published_ids):
                    avg_corners = np.mean(self.coordinate_buffers[marker_id], axis=0).reshape((4, 2))

                    msg = Float32MultiArray()
                    msg.data = [float(marker_id)] + [c for p in avg_corners for c in p]

                    self.aruco_pub_detection.publish(msg)
                    rospy.loginfo(f"Published ArUco detection: {msg.data}")

                    # Clear buffer and mark as published (one-shot behaviour)
                    self.coordinate_buffers[marker_id].clear()
                    self.published_ids.add(marker_id)

        return frame

    def publish_to_ros(self, frame):
        ok, enc = cv2.imencode('.jpg', frame)
        if not ok:
            rospy.logwarn("JPEG encode failed")
            return
        msg_out = CompressedImage()
        msg_out.header.stamp = rospy.Time.now()
        msg_out.format = "jpeg"
        msg_out.data = enc.tobytes()
        self.aruco_pub_img.publish(msg_out)

def main():
    rospy.init_node('aruco_detector', anonymous=True)
    rospy.loginfo("Node 'aruco_detector' started")
    detector = ArucoDetector()
    rospy.spin()

if __name__ == "__main__":
    main()
