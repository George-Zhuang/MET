from __future__ import annotations

import carb
import omni.usd
from omni.anim.people.scripts.global_agent_manager import GlobalAgentManager


class ORCAgentManager:
    """Global class which stores current and predicted positions of all characters and moving objects."""

    __instance: ORCAgentManager = None

    def __init__(self):
        if self.__instance is not None:
            raise RuntimeError("Only one instance of GlobalCharacterPositionManager is allowed")
        # character dict that match character object with prim path
        ORCAgentManager.__instance = self

    def destroy(self):
        ORCAgentManager.__instance = None

    def __del__(self):
        self.destroy()

    @classmethod
    def get_instance(cls) -> ORCAgentManager:
        if cls.__instance is None:
            ORCAgentManager()
        return cls.__instance

    def inject_command_for_all_agents(self, command_list, force):
        # if Global Character Manager is none, if so, remind user to check whether simulation has been started
        if not GlobalAgentManager.has_instance():
            carb.log_warn("Global Character Manager is None. Please check out whether simulation has already start")
        GlobalAgentManager.get_instance().inject_command_for_all_agents(command_list, force)

    def inject_command(self, agent_prim_path, command_list, force=False):
        # if Global Character Manager is none, if so, remind user to check whether simulation has been started
        if not GlobalAgentManager.has_instance():
            carb.log_warn("Global Character Manager is None. Please check out whether simulation has already start")
        GlobalAgentManager.get_instance().inject_command(agent_prim_path, command_list, force=force)
