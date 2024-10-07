import io
import json
from pathlib import PurePosixPath

import carb
import omni.client
import omni.ext
import omni.kit.app
import yaml

from .extension import get_ext_id


class FileUtil:
    def get_parent_path(file_path):
        url = omni.client.break_url(file_path)
        parent_path = PurePosixPath(url.path).parent
        parent_url = omni.client.make_url(
            scheme=url.scheme,
            host=url.host,
            path=str(parent_path),
        )
        return parent_url.replace("file:/", "").replace("file:", "")

    def get_absolute_path(base_path, file_path):
        return (
            omni.client.utils.make_absolute_url_if_possible(base_path, file_path)
            .replace("file:/", "")
            .replace("file:", "")
        )


class YamlFileUtil:
    def load_yaml(file_path):
        result, version, context = omni.client.read_file(file_path)
        if result != omni.client.Result.OK:
            carb.log_error("Cannot load YAML file at this path: " + file_path)
            return None
        yaml_str = memoryview(context).tobytes().decode("utf-8")
        return yaml.safe_load(yaml_str)

    async def load_yaml_async(file_path):
        result, version, context = await omni.client.read_file_async(file_path)
        if result != omni.client.Result.OK:
            carb.log_error("Cannot load YAML file at this path: " + file_path)
            return None
        yaml_str = memoryview(context).tobytes().decode("utf-8")
        return yaml.safe_load(yaml_str)

    def save_yaml(file_path, yanml_data):
        stream = io.StringIO("")
        yaml.dump(yanml_data, stream, sort_keys=False)
        result = omni.client.write_file(url=file_path, content=stream.getvalue().encode("utf-8"))
        if result != omni.client.Result.OK:
            carb.log_warn("omni.client result is not OK")
            return False
        carb.log_info("save yaml file at: " + str(file_path))
        return True


class TextFileUtil:
    def create_text_file(file_path, content_str=None):
        try:
            file = open(file_path, "w")
            file.seek(0)
            file.truncate()  # Remove previous content if it exists
            file.close()
            if content_str:
                return TextFileUtil.write_text_file(file_path, content_str)
            else:
                return True
        except IOError:
            return False

    def is_text_file_exist(file_path):
        result, version, context = omni.client.read_file(file_path)
        return result == omni.client.Result.OK

    def read_text_file(file_path):
        result, version, context = omni.client.read_file(file_path)
        if result != omni.client.Result.OK:
            carb.log_error("Cannot load TXT file at this path: " + str(file_path))
            return None
        return memoryview(context).tobytes().decode("utf-8")

    def write_text_file(file_path, contentStr):
        if contentStr == None:
            return False
        result = omni.client.write_file(file_path, contentStr.encode("utf-8"))
        if result != omni.client.Result.OK:
            carb.log_warn("omni.client result is not OK")
            return False
        return True

    def copy_text_file(source_file, target_file):
        content_str = TextFileUtil.read_text_file(source_file)
        if not TextFileUtil.create_text_file(target_file):
            return False
        return TextFileUtil.write_text_file(target_file, content_str)


class JSONFileUtil:
    def load_from_file(file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data

    def write_to_file(file_path, data):
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4, separators=(",", ": "))
