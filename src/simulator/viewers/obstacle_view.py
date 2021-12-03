class ObstacleView:
    def __init__(self, viewer, obstacle):
        self.viewer = viewer
        self.obstacle = obstacle

    def draw_obstacle_to_frame(self):
        obstacle = self.obstacle

        # grab the obstacle pose
        obstacle_pos, obstacle_theta = obstacle.pose.vunpack()

        # draw the obstacle to the frame
        obstacle_poly = obstacle.global_geometry.vertices
        self.viewer.current_frame.add_polygons(
            [obstacle_poly], color="dark red", alpha=0.4
        )

        # === FOR DEBUGGING: ===
        # self._draw_bounding_circle_to_frame()

    def _draw_bounding_circle_to_frame(self):
        c, r = self.obstacle.global_geometry.bounding_circle
        self.viewer.current_frame.add_circle(pos=c, radius=r, color="black", alpha=0.2)
