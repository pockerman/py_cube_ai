import cv2
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from std_msgs.msg import String


NODE_NAME: str = 'opencv_canny_edge_detector'
TOPIC_NAME: str = "opencv_video_frames"
CAMERA_IDX: int = 0
IMAGES_QUEUE_SIZE = 10

class OpenCVCannyEdgeDetectorNode(Node):
    def __init__(self, topic_name: str=TOPIC_NAME,
                 images_queue_size: int = IMAGES_QUEUE_SIZE,
                 threshold1: int=100, threshold2: int=100):
        super().__init__(NODE_NAME)
        
        self.threshold1 = threshold1
        self.threshold2 = threshold2

        self.subscription = self.create_subscription(Image, topic_name, self.detect, images_queue_size)
        self.br = CvBridge()
        

    def detect(self, msg):
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(msg)

        canny_edges = cv2.Canny(image=current_frame, 
                                threshold1=self.threshold1, 
                                threshold2=self.threshold2)

        canny_img = PILImage.fromarray(canny_edges)
        canny_img.show()

        cv2.waitKey(1)

        

def main(args=None):


    # Initialize the rclpy library
    rclpy.init(args=args)
   
    # Create the node
    edge_detector = OpenCVCannyEdgeDetectorNode()
    edge_detector.get_logger().info('Running OpenCVCannyEdgeDetectorNode. Ctrl+C to stop.')
   
    # Spin the node so the callback function is called.
    rclpy.spin(edge_detector)
   
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()
       
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()