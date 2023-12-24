"""
ROS2 program to publish real-time streaming  video from your built-in webcam
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import cv2 

NODE_NAME: str = "opencv_video_publisher"
TOPIC_NAME: str ="opencv_video_frames"
IMAGES_QUEUE_SIZE = 10
PUBLISH_MESSAGE_RATE: float = 5.0

  
class OpenCVVideoPublisher(Node):
  """
  Create an ImagePublisher class, which is a subclass of the Node class.
  """
  
  def __init__(self, topic_name: str=TOPIC_NAME, 
  	images_queue_size: int = IMAGES_QUEUE_SIZE,
  	publish_message_rate: float=PUBLISH_MESSAGE_RATE):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__(NODE_NAME)
       
    # Create the publisher. This publisher will publish an Image
    # to the video_frames topic. The queue size is 10 messages.
    self.publisher_ = self.create_publisher(Image, topic_name, images_queue_size)
       
    # We will publish a message every 0.1 seconds
    timer_period = 0.1  # seconds
       
    # Create the timer
    self.timer = self.create_timer(publish_message_rate, self.timer_callback)
          
    # Create a VideoCapture object
    # The argument '0' gets the default webcam.
    self.cap = cv2.VideoCapture(0)
          
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    self.get_logger().info(f'Finished {NODE_NAME} initialization')
    self.get_logger().info(f'{NODE_NAME} publishes on topic {topic_name}')
    self.get_logger().info(f'{NODE_NAME} images_queue_size {images_queue_size}')
    self.get_logger().info(f'{NODE_NAME} publish message rate {publish_message_rate}')
    
  def timer_callback(self):
    """
    Callback function.
    This function gets called every 0.1 seconds.
    """

    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame = self.cap.read()
           
    if ret == True:
      # Publish the image.
      # The 'cv2_to_imgmsg' method converts an OpenCV
      # image to a ROS 2 image message
      self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
  
    # Display the message on the console
    self.get_logger().info('Publishing video frame')
   
def main(args=None):
   
  # Initialize the rclpy library
  rclpy.init(args=args)
   
  # Create the node
  image_publisher = OpenCVVideoPublisher(topic_name=TOPIC_NAME,
  	                                     images_queue_size=IMAGES_QUEUE_SIZE,
  	                                     publish_message_rate=PUBLISH_MESSAGE_RATE)
   
  # Spin the node so the callback function is called.
  rclpy.spin(image_publisher)
   
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_publisher.destroy_node()
   
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()
