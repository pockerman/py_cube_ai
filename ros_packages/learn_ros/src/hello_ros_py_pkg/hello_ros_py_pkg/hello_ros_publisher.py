import rclpy
from rclpy.node import Node

from std_msgs.msg import String


NODE_NAME: str = 'hello_ros_publisher'

class HelloROSPublisher(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        self.publisher_ = self.create_publisher(String, 'hello_ros_topic', 10)

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %i' % self.get_clock().now().nanoseconds
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    hello_ros_pub_node = HelloROSPublisher()

    hello_ros_pub_node.get_logger().info('Running HelloRosPublisher. Stop: Ctrl+C')

    rclpy.spin(hello_ros_pub_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()