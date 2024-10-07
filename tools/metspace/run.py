# import argparse
import glob
import os
import re
from multiprocessing import Process

CUSTOM_APP_PATH = ""
from isaacsim import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 1920, "height": 1080}

class METSpace:
    def __init__(self, sim_app, num_runs=1):
        self.num_runs = num_runs
        self.config_dict = None
        self._sim_manager = None
        self._data_generator = None
        self._settings = None
        self._sim_app = sim_app
        self._nav_mesh_event_handle = None
        self.navmesh_baking_complete = False

    def set_simulation_settings(self):
        import carb
        import omni.replicator.core as rep

        rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
        self._settings = carb.settings.get_settings()
        self._settings.set("/rtx/rtxsensor/coordinateFrameQuaternion", "0.5,-0.5,-0.5,-0.5")
        self._settings.set("/app/scripting/ignoreWarningDialog", True)
        self._settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        self._settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        self._settings.set("/persistent/exts/metspace/aim_camera_to_character", True)
        self._settings.set("/persistent/exts/metspace/min_camera_distance", 6.5)
        self._settings.set("/persistent/exts/metspace/max_camera_distance", 14.5)
        self._settings.set("/persistent/exts/metspace/max_camera_look_down_angle", 60)
        self._settings.set("/persistent/exts/metspace/min_camera_look_down_angle", 0)
        self._settings.set("/persistent/exts/metspace/min_camera_height", 2)
        self._settings.set("/persistent/exts/metspace/max_camera_height", 3)
        self._settings.set("/persistent/exts/metspace/character_focus_height", 0.7)
        self._settings.set("/persistent/exts/metspace/frame_write_interval", 1)

        self._settings.set("/persistent/isaac/asset_root/default", "/isaac-sim/Assets/Isaac/4.0")

    def bake_navmesh(self):
        import carb
        import omni.anim.navigation.core as nav

        _nav = nav.nav.acquire_interface()
        # Do not proceed if navmesh volume does not exist
        if _nav.get_navmesh_volume_count() == 0:
            carb.log_error("Scene does not have navigation volume. Stoping data generation and closing app.")
            self._sim_app.update()
            self._sim_app.close()
            return

        _nav.start_navmesh_baking()

        def nav_mesh_callback(event):
            if event.type == nav.EVENT_TYPE_NAVMESH_READY:
                self._nav_mesh_event_handle = None
                self.navmesh_baking_complete = True
            elif event.type == nav.EVENT_TYPE_NAVMESH_BAKE_FAILED:
                carb.log_error("Navmesh baking failed. Stoping data generation and closing app.")
                self._nav_mesh_event_handle = None
                self._sim_app.update()
                self._sim_app.close()

        self._nav_mesh_event_handle = _nav.get_navmesh_event_stream().create_subscription_to_pop(nav_mesh_callback)

    def setup(self, config_file):
        import carb
        from omni.isaac.core.utils.stage import open_stage
        from metspace.core.simulation import SimulationManager
        self.set_simulation_settings()
        self._sim_manager = SimulationManager()
        self.config_dict, is_modified = self._sim_manager.load_config_file(config_file)
        # load env and bake navmesh
        stage_open_result = open_stage(self.config_dict["scene"]["asset_path"])
        if not stage_open_result:
            carb.log_error("Unable to open stage {}".format(self.config_dict["scene"]["asset_path"]))
            self._sim_app.close()
        self._sim_app.update()
        self.bake_navmesh()
        while self.navmesh_baking_complete != True:
            self._sim_app.update()
        # Create character, robot and cameras
        self._sim_manager.load_agents_cameras_from_config_file()
        self._sim_app.update()


def enable_extensions():
    # Enable extensions
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.kit.viewport.window")
    enable_extension("omni.kit.manipulator.prim")
    enable_extension("omni.kit.property.usd")
    enable_extension("omni.anim.navigation.bundle")
    enable_extension("omni.anim.timeline")
    enable_extension("omni.anim.graph.bundle")
    enable_extension("omni.anim.graph.core")
    enable_extension("omni.anim.retarget.bundle")
    enable_extension("omni.anim.retarget.core")
    enable_extension("omni.kit.scripting")
    enable_extension("omni.extended.materials")
    enable_extension("omni.anim.people")
    enable_extension("omni.kit.mesh.raycast")
    enable_extension("metspace.core")

def run(config_file, num_runs=1):

    # Initalize kit app
    kit = SimulationApp(launch_config=CONFIG, experience=CUSTOM_APP_PATH)

    # Enable extensions
    enable_extensions()
    kit.update()

    # Load modules from extensions
    import carb
    import omni.kit.loop._loop as omni_loop

    loop_runner = omni_loop.acquire_loop_interface()
    loop_runner.set_manual_step_size(1.0 / 30.0)
    loop_runner.set_manual_mode(True)
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", False)

    while kit.is_running():
        my_world.step(render=True)
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                reset_needed = False
            if i >= 0 and i < 1000:
                # forward
                # my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[2, 0]))
                print(my_jetbot.get_linear_velocity())
            elif i >= 1000 and i < 1300:
                # rotate
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, np.pi / 12]))
                print(my_jetbot.get_angular_velocity())
            elif i >= 1300 and i < 2000:
                # forward
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.05, 0]))
            elif i == 2000:
                i = 0
            i += 1
        if args.test is True:
            break


    simulation_app.close()
