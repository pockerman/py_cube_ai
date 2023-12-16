import rclpy
from rclpy.node import Node

from std_msgs.msg import String


NODE_NAME: str = 'hello_ros_subscriber'

class HelloROSSubscriberer(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.subscription_ = self.create_subscription(String,
                                                      'hello_ros_topic',
                                                      self.sub_callback,
                                                      10)

    def sub_callback(self, msg):
        self.get_logger().info('Received message: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    my_ros_sub_node = HelloROSSubscriberer()

    my_ros_sub_node.get_logger().info('Running HelloROSSubscriberer. Ctrl+C to stop.')
    rclpy.spin(my_ros_sub_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()