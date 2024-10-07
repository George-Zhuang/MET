import asyncio

import carb
import carb.events
import omni.kit.ui
import omni.replicator.core as rep
import omni.timeline
from omni.isaac.core.utils import prims, semantics
from omni.replicator.core import orchestrator

from .settings import Settings
from .stage_util import CameraUtil, CharacterUtil, LidarCamUtil, RobotUtil
from .scripts.track_manager import TrackManager

MAX_STEP_ATTEMPTS = 10


class DataGeneration:
    """
    Class to handle all components of data generation - creating render products, registering writers,
    starting and stopping data generation.
    """

    def __init__(self, yaml_data=None, camera_start_index=0, lidar_start_index=0):
        self.writer_name = ""
        self.writer_params = {}
        self._writer = None
        self._write_robot_data = False
        self._num_frames = 0
        self._control_timeline = True
        self._orchestrator_status = rep.orchestrator.get_status()
        self._in_running_state = False

        # Additional config
        self._camera_start_index = camera_start_index
        self._lidar_start_index = lidar_start_index

        self._data_generation_done_callback = None

        # Orchestrator status update callback
        self._orchestrator_status_cb = rep.orchestrator.register_status_callback(self._on_orchestrator_status_changed)

        # Stage event callback
        self._sub_stage_event = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop_by_type(int(omni.usd.StageEventType.CLOSING), self._on_stage_closing_event)
        )

        self._num_cameras = 0
        self._num_lidars = 0
        self._render_products = []
        self._camera_list = []
        self._lidar_list = []
        self._camera_path_list = []
        self._lidar_path_list = []

        # Viewport Information
        self._show_grid = None
        self._show_outline = None
        self._show_navMesh = None
        self._show_camera = None
        self._show_light = None
        self._show_audio = None
        self._show_skeleton = None
        self._show_meshes = None

        if yaml_data:
            self.load_yaml_config(yaml_data)

    def load_yaml_config(self, yaml_data):
        stage = omni.usd.get_context().get_stage()
        if yaml_data["global"]["simulation_length"]:
            self._num_frames = yaml_data["global"]["simulation_length"] * 30
            self._num_frames += Settings.extend_data_generation_length() * 30
        if "camera_num" in yaml_data["global"]:
            self._num_cameras = yaml_data["global"]["camera_num"]
        if "lidar_num" in yaml_data["global"]:
            self._num_lidars = yaml_data["global"]["lidar_num"]
        if "camera_list" in yaml_data["global"]:
            self._camera_path_list = yaml_data["global"]["camera_list"]
        if "lidar_list" in yaml_data["global"]:
            self._lidar_path_list = yaml_data["global"]["lidar_list"]
        if yaml_data["robot"]["write_data"]:
            self._write_robot_data = yaml_data["robot"]["write_data"]
        if yaml_data["replicator"]["writer"]:
            self.writer_name = yaml_data["replicator"]["writer"]
        if yaml_data["replicator"]["parameters"]:
            self.writer_params = yaml_data["replicator"]["parameters"]

    async def start_recorder_sync(self):
        if self._orchestrator_status is orchestrator.Status.STOPPED:
            if self._init_recorder():
                num_frames = None if self._num_frames <= 0 else self._num_frames
                skip_frames = carb.settings.get_settings().get(
                    "/persistent/exts/metspace/skip_starting_frames"
                )
                carb.settings.get_settings().set("/omni/replicator/captureOnPlay", True)
                timeline = omni.timeline.get_timeline_interface()

                # Call replicator step until replicator starts timeline
                attempts = 0
                while attempts < MAX_STEP_ATTEMPTS:
                    attempts += 1
                    await rep.orchestrator.step_async(pause_timeline=False)
                    if timeline.is_playing():
                        break

                if attempts >= MAX_STEP_ATTEMPTS:
                    carb.log_error(
                        "SDG stopped, replicator is unable to start timeline. Try running SDG in async mode."
                    )
                    rep.orchestrator.stop()
                    self._clear_recorder()
                    self._in_running_state = False
                    return

                # In the number of frames to generate
                for i in range(num_frames + skip_frames - attempts):
                    if timeline.is_playing():
                        await rep.orchestrator.step_async(pause_timeline=False)
                    else:
                        carb.log_warn("SDG stopped prematurely.")
                        break
                timeline.stop()

            # Cleanup if init failed or if data generation is complete
            rep.orchestrator.stop()
            self._clear_recorder()
            self._in_running_state = False
        else:
            carb.log_warn(
                f"Replicator's current state({self._orchestrator_status.name}) is different state than STOPPED. Try again in a bit."
            )

    async def start_recorder_async(self):
        if self._orchestrator_status is orchestrator.Status.STOPPED:
            if self._init_recorder():
                num_frames = None if self._num_frames <= 0 else self._num_frames
                skip_frames = carb.settings.get_settings().get(
                    "/persistent/exts/metspace/skip_starting_frames"
                )

                await rep.orchestrator.run_async(
                    num_frames=num_frames + skip_frames, start_timeline=self._control_timeline
                )
                self._in_running_state = True
            else:
                self._clear_recorder()
        else:
            carb.log_warn(
                f"Replicator's current state({self._orchestrator_status.name}) is different state than STOPPED. Try again in a bit."
            )

    async def stop_recorder_async(self):
        if self._orchestrator_status is orchestrator.Status.STARTED:
            await rep.orchestrator.stop_async()
            if self._control_timeline:
                await self._set_timeline_state_async(case="reset")
            self._clear_recorder()
            self._in_running_state = False
        else:
            carb.log_warn(
                f"Replicator's current state({self._orchestrator_status.name}) is different state than STARTED. Try again in a bit."
            )

    def run_until_complete(self):
        num_frames = None if self._num_frames <= 0 else self._num_frames
        rep.orchestrator.run_until_complete(num_frames=num_frames, start_timeline=self._control_timeline)
        self._in_running_state = True

    def _init_recorder(self) -> bool:
        if self._writer is None:
            try:
                self._writer = rep.WriterRegistry.get(self.writer_name)
            except Exception as e:
                carb.log_error(f"Could not create writer {self.writer_name}: {e}")
                return False
        try:
            self._writer.initialize(**self.writer_params)
        except Exception as e:
            carb.log_error(f"Could not initialize writer {self.writer_name}: {e}")
            return False

        # Fetch from stage if config file did not specify a camera list
        if not self._camera_path_list:
            self._camera_list = self._get_camera_list(self._num_cameras, self._camera_start_index)
            self._camera_path_list = [cam.GetPrimPath() for cam in self._camera_list]
            self._lidar_list = self._get_lidar_list(self._num_lidars, self._lidar_start_index)
            if self._lidar_list:
                self._lidar_path_list = [ldr.GetPrimPath() for ldr in self._lidar_list]
        self._render_products = self.create_render_product_list()

        # Hide debugging visualizations like navmesh and grid in the viewport
        hide_visualization = carb.settings.get_settings().get(
            "/persistent/exts/metspace/hide_visualization"
        )
        if hide_visualization:
            self.store_viewport_settings()
            self.set_viewport_settings()

        if not self._render_products:
            carb.log_error("No valid render products found to initialize the writer.")
            return False

        try:
            self._writer.attach(self._render_products)
        except Exception as e:
            carb.log_error(f"Could not attach render products to writer: {e}")
            return False

        # Temporary solution: attach timeline to Replicator before recording
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", True)
        return True

    def _on_orchestrator_status_changed(self, status):
        new_status = status is not self._orchestrator_status
        if new_status:
            self._orchestrator_status = status
            # Check if the recorder was running and it stopped because it reached the number of requested frames
            has_finished_recording = self._in_running_state and status is rep.orchestrator.Status.STOPPED
            if has_finished_recording:
                asyncio.ensure_future(self._on_orchestrator_finish_async())

    async def _on_orchestrator_finish_async(self):
        if self._control_timeline:
            await self._set_timeline_state_async(case="reset")
        await rep.orchestrator.wait_until_complete_async()
        self._clear_recorder()
        self._in_running_state = False

    async def _set_timeline_state_async(self, case="reset"):
        timeline = omni.timeline.get_timeline_interface()
        if case == "reset":
            if timeline.is_playing():
                timeline.stop()
            timeline.set_current_time(0)
            await omni.kit.app.get_app().next_update_async()
        elif case == "pause":
            if timeline.is_playing():
                timeline.pause()
        elif case == "resume":
            if not timeline.is_playing():
                timeline.play()

    def _clear_recorder(self):
        if self._writer:
            self._writer.detach()
            self._writer = None
        for rp in self._render_products:
            rp.destroy()
        self._render_products.clear()

        # Recover viewport state if debugging visualizations were automatically hidden in data generation
        hide_visualization = carb.settings.get_settings().get(
            "/persistent/exts/metspace/hide_visualization"
        )
        if hide_visualization:
            self.recover_viewport_settings()

        # Unsubscribe events
        self._sub_stage_event = None
        self._orchestrator_status_cb.unregister()
        # Temporary solution: detach Replicator from timeline when recording finish
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        # Done callback
        if self._data_generation_done_callback:
            self._data_generation_done_callback()

    def _on_stage_closing_event(self, e: carb.events.IEvent):
        if self._orchestrator_status is not orchestrator.Status.STOPPED:
            rep.orchestrator.stop()
        self._clear_recorder()

    def _on_editor_quit_event(self, e: carb.events.IEvent):
        # Fast shutdown of the extension, stop recorder save config files
        if self._orchestrator_status is not orchestrator.Status.STOPPED:
            rep.orchestrator.stop()
            self._clear_recorder()

    def on_shutdown(self):
        # Clean shutdown of the extension, called when the extension is unloaded (not called when the editor is closed)
        if self._orchestrator_status is not orchestrator.Status.STOPPED:
            rep.orchestrator.stop()
            self._clear_recorder()
        self._orchestrator_status_cb.unregister()

    def register_recorder_done_callback(self, fn: callable):
        self._data_generation_done_callback = fn

    def store_viewport_settings(self):
        self._show_grid = carb.settings.get_settings().get("/app/viewport/grid/enabled")
        self._show_outline = carb.settings.get_settings().get("/app/viewport/outline/enabled")
        self._show_navMesh = carb.settings.get_settings().get(
            "/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh"
        )
        self._show_camera = carb.settings.get_settings().get("/app/viewport/show/cameras")
        self._show_light = carb.settings.get_settings().get("/app/viewport/show/lights")
        self._show_audio = carb.settings.get_settings().get("/app/viewport/show/audio")
        self._show_skeleton = carb.settings.get_settings().get("/app/viewport/usdcontext-/scene/skeletons/visible")
        self._show_meshes = carb.settings.get_settings().get("/app/viewport//usdcontext-/scene/meshes/visible")

    def set_viewport_settings(self):
        carb.settings.get_settings().set("/app/viewport/grid/enabled", False)
        carb.settings.get_settings().set("/app/viewport/outline/enabled", False)
        carb.settings.get_settings().set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        carb.settings.get_settings().set("/app/viewport/show/cameras", False)
        carb.settings.get_settings().set("/app/viewport/show/lights", False)
        carb.settings.get_settings().set("/app/viewport/show/audio", False)
        carb.settings.get_settings().set("/app/viewport/usdcontext-/scene/skeletons/visible", False)
        carb.settings.get_settings().set("/app/viewport/usdcontext-/scene/meshes/visible", True)

    def recover_viewport_settings(self):
        if self._show_grid != None:
            carb.settings.get_settings().set("/app/viewport/grid/enabled", self._show_grid)
        if self._show_outline != None:
            carb.settings.get_settings().set("/app/viewport/outline/enabled", self._show_outline)
        if self._show_navMesh != None:
            carb.settings.get_settings().set(
                "/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", self._show_navMesh
            )
        if self._show_camera != None:
            carb.settings.get_settings().set("/app/viewport/show/cameras", self._show_camera)
        if self._show_light != None:
            carb.settings.get_settings().set("/app/viewport/show/lights", self._show_light)
        if self._show_audio != None:
            carb.settings.get_settings().set("/app/viewport/show/audio", self._show_audio)
        if self._show_skeleton != None:
            carb.settings.get_settings().set("/app/viewport/usdcontext-/scene/skeletons/visible", self._show_skeleton)
        if self._show_meshes != None:
            carb.settings.get_settings().set("/app/viewport//usdcontext-/scene/meshes/visible", self._show_meshes)

    def create_render_product_list(self):
        render_product_list = []
        for path in self._camera_path_list:
            render_product_list.append(rep.create.render_product(path, (1280, 720)))
        if self._write_robot_data:
            # Add cameras as render product
            # for c in RobotUtil.get_n_robot_cameras(2):
            for c in TrackManager.get_instance().get_agent_cameras():
                cam_path = prims.get_prim_path(c)
                rp = rep.create.render_product(cam_path, resolution=(256, 256))
                render_product_list.append(rp)
            # Add lidars as render products
            # for l in RobotUtil.get_robot_lidar_cameras():
            #     cam_path = prims.get_prim_path(l)
            #     rp = rep.create.render_product(cam_path, resolution=[1, 1])
            #     render_product_list.append(rp)
        for path in self._lidar_path_list:
            lidar_rp = rep.create.render_product(path, resolution=[1, 1])
            render_product_list.append(lidar_rp)
        return render_product_list

    def _get_camera_list(self, num_cameras, camera_start_index=0):
        camera_list = []
        cameras_in_stage = CameraUtil.get_cameras_in_stage()
        # If num_cam == -1, we honor whatever cameras are in the stage
        if num_cameras == -1:
            return cameras_in_stage
        camera_end_index = camera_start_index + num_cameras
        if camera_end_index > len(cameras_in_stage):
            camera_end_index = len(cameras_in_stage)
            carb.log_warn(
                "Camera Number is greater than the cameras in the stage. Only the cameras in the stage will have output. Please click Set Up Simulation before Data Generation."
            )
        camera_list = cameras_in_stage[camera_start_index:camera_end_index]
        return camera_list

    def _get_lidar_list(self, num_lidar, lidar_start_index=0):
        # fetching all valid lidar prims in the stage
        lidar_list = LidarCamUtil.get_valid_lidar_cameras_in_stage()
        # If num_lidar == -1, we honor whatever cameras are in the stage
        if num_lidar == -1:
            return lidar_list
        # Warn uses about the invalid lidars in stage
        invalid_lidars_in_stage = [
            lidar for lidar in LidarCamUtil.get_lidar_cameras_in_stage() if lidar not in lidar_list
        ]
        if len(invalid_lidars_in_stage) > 0:
            carb.log_warn("Invalid lidar camera found in stage: " + str(invalid_lidars_in_stage))

        # Clamp the start index
        lidar_start_index = max(lidar_start_index, 0)
        lidar_start_index = min(lidar_start_index, len(lidar_list))

        lidar_end_index = lidar_start_index + num_lidar
        if lidar_end_index > len(lidar_list):
            lidar_end_index = len(lidar_list)
            carb.log_warn(
                "Lidar Number is greater than the valid lidars in the stage. Only the lidars in the stage will have output. Please click Set Up Simulation before Data Generation."
            )
        lidar_list = lidar_list[lidar_start_index:lidar_end_index]
        return lidar_list

    def _lidar_fusion_renderproduct_prune(self):
        matched_cameras = []
        matched_lidars = []
        unmatched = []
        for camera in self._camera_list:
            camera_name = CameraUtil.get_camera_name_without_prefix(camera)
            has_match = False
            for lidar in self.lidar_list:
                lidar_name = LidarCamUtil.get_lidar_name_without_prefix(lidar)
                if camera_name == lidar_name:
                    has_match = True
                    break
            if has_match:
                matched_cameras.append(camera)
            else:
                unmatched.append(camera)
        self._camera_list = matched_cameras

        for lidar in self.lidar_list:
            lidar_name = LidarCamUtil.get_lidar_name_without_prefix(lidar)
            has_match = False
            for camera in self._camera_list:
                camera_name = CameraUtil.get_camera_name_without_prefix(camera)
                if lidar_name == camera_name:
                    has_match = True
                    break
            if has_match:
                matched_lidars.append(lidar)
            else:
                unmatched.append(lidar)
        self.lidar_list = matched_lidars

        if unmatched:
            carb.log_error(str(unmatched) + " have no matching cameras in the stage, and thus will not have any output")

    def add_ppl_semantics(character_prim_list):
        stage = omni.usd.get_context().get_stage()
        for character_prim_path in character_prim_list:
            character_prim = stage.GetPrimAtPath(character_prim_path)
            character_name = CharacterUtil.get_character_name(character_prim)
            semantics.add_update_semantics(character_prim, semantic_label=character_name)
