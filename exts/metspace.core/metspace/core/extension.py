import asyncio

import carb
import omni.ext
import omni.kit.app

from .settings import AssetPaths

_extension_instance = None
_ext_id = None
_ext_path = None


def get_instance():
    return _extension_instance


def get_ext_id():
    return _ext_id


def get_ext_path():
    return _ext_path


class Main(omni.ext.IExt):

    ext_version = ""

    def on_startup(self, ext_id):
        print("[metspace.Core] Extension startup")
        # https://jirasw.nvidia.com/browse/METROPERF-299
        import warp

        warp.init()

        # Set instance
        global _extension_instance
        _extension_instance = self
        global _ext_id
        _ext_id = ext_id
        global _ext_path
        _ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)

        # Get extension version
        Main.ext_version = str(ext_id).split("-")[-1]
        # Handle async startup tasks
        self._is_startup_async_done = False
        self._startup_task = asyncio.ensure_future(self.startup_async())
        self._startup_task.add_done_callback(self.startup_async_done)

    def on_shutdown(self):
        print("[metspace.Core] Extension shutdown")
        global _extension_instance
        _extension_instance = None
        global _ext_id
        _ext_id = None
        global _ext_path
        _ext_path = None

    async def startup_async(self):
        """
        Async startup tasks for this extension.
        """
        # Get asset paths from nucleus server
        await AssetPaths.cache_paths_async()

    def startup_async_done(self, context):
        print("[metspace.Core] Extension startup async done")
        self._is_startup_async_done = True
        self._startup_task = None  # Release handle

    def check_startup_async_done(self):
        return self._is_startup_async_done

    def add_startup_async_done_callback(self, callback: callable):
        if self._startup_task:
            self._startup_task.add_done_callback(callback)

    def remove_startup_async_done_callback(self, callback: callable):
        if self._startup_task:
            self._startup_task.remove_done_callback(callback)
