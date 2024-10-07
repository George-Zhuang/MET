# --------------------------------------------------------------------
# MultiModal Embodied Tracking
#  - Run models to get the action
# --------------------------------------------------------------------
import carb
import numpy as np
import omni.replicator.core as rep
from PIL import Image

class Robot_policy:
    """
    Command class to  action
    """

    def __init__(self, robot, controller, camera_prim, model,
                 linear_speed: float = 1.5,
                 angular_speed: float = 0.350, # 0.350 rad/s = 20 deg/s
                 duration: float = 5,
                 inference_rate: int = 6,
                 resolution: list = [256, 256],
                 frame_rate: int = 30
                 ):
        self.robot = robot
        self.controller = controller
        self.camera_prim = camera_prim
        self.model = model
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.duration = duration
        self.action_time = 1 / inference_rate
        
        self.time_elapsed = 0
        self.time_elapsed_update = 0
        self.is_setup = False
        self.finished = False
        self.command = [0, 0]

        camera_rep = rep.create.render_product(
            str(self.camera_prim.GetPrimPath()), resolution=resolution
        )
        self.camera_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.camera_rgb.attach(camera_rep)
        self.camera_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.camera_depth.attach(camera_rep)

    def setup(self):
        self.time_elapsed = 0
        self.is_setup = True

    def exit_command(self):
        self.is_setup = False
        return True

    def update(self, dt):
        self.time_elapsed += dt
        self.time_elapsed_update += dt
        if self.time_elapsed_update > self.action_time:
            # random image
            self.forward_policy()
            self.time_elapsed_update = 0
        self.robot.apply_wheel_actions(
            self.controller._open_loop_wheel_controller.forward(self.command)
        )

    def execute(self, dt):
        if self.finished:
            return True

        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def force_quit_command(self):
        position, orientation = self.robot.get_world_pose()
        self.robot.apply_wheel_actions(
            self.controller.forward(start_position=position, start_orientation=orientation, goal_position=position)
        )
        self.is_setup = False
        self.finished = True
        return
    
    def forward_policy(self):
        """
        Get the action from the model
            stop = 0
            move_forward = 1
            turn_left = 2
            turn_right = 3
        """
        # get image
        # rep.orchestrator.step_async()
        image = self.camera_rgb.get_data()[:, :, :3]
        depth = self.camera_depth.get_data()
        carb.log_warn(f"image.shape: {image.shape}")
        carb.log_warn(f"depth.shape: {depth.shape}")
        # save the image for debugging
        image = Image.fromarray(image)
        image.save(f"F:/2024_EVT/EVT/evt_generation/tmp/camera_debug/image_{self.time_elapsed}.png")
        # save the depth for debugging
        depth = (depth / np.max(depth) * 255).astype(np.uint8)
        depth = Image.fromarray(depth)
        # normalize the depth to unit8
        depth.save(f"F:/2024_EVT/EVT/evt_generation/tmp/camera_debug/depth_{self.time_elapsed}.png")

        action = self.model.predict(image, depth=depth)

        
        if action == 0:
            self.command = [0, 0]
        elif action == 1:
            self.command = [self.linear_speed, 0]
        elif action == 2:
            self.command = [0, self.angular_speed]
        elif action == 3:
            self.command = [0, -self.angular_speed]
        else:
            raise ValueError(f"Invalid action: {action}, must be in [0, 1, 2, 3], where 0=stop, 1=move_forward, 2=turn_left, 3=turn_right")

        return action
    