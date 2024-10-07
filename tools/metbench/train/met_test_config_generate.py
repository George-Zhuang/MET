import os
import random
import yaml

train_scene_paths = [
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_01.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_02.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_03.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_04.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_05.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_06.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_07.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Hospital/hospital_08.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_01.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_02.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_03.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_04.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_05.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_06.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_07.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_08.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_09.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_10.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_11.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_12.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_13.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_14.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_15.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_16.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_17.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_18.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_19.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_20.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_21.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_22.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_23.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_24.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_25.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_26.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_27.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_28.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_29.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_30.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_31.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_32.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Outdoor/Rivermark/rivermark_navmesh_34.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/full_warehouse_01.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/full_warehouse_02.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/full_warehouse_03.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/full_warehouse_04.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/full_warehouse_05.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/full_warehouse_06.usd",
    "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/warehouse_01.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves_01.usd",
    # "/isaac-sim/Assets/Isaac/4.0/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts_01.usd",
]

default_config_path = "/isaac-sim/tools/met/config_test1.yaml"
default_config = yaml.safe_load(open(default_config_path, "r"))
config_save_dir = "/isaac-sim/tmp/mettest/configs"
os.makedirs(config_save_dir, exist_ok=True)
num_seed_per_scene = 5
robot_types = ["nova_carter", "transporter", "carter_v1", "dingo", "jetbot"]

for scene_path in train_scene_paths:
    for i in range(num_seed_per_scene):
        config = default_config.copy()
        # get a random seed
        seed = random.randint(0, 1000)
        config["met"]["global"]["seed"] = seed
        # set random seed
        random.seed(seed)
        config["met"]["scene"]["asset_path"] = scene_path
        # character num between 1 to 20
        character_num = random.randint(1, 20)
        config["met"]["character"]["num"] = character_num
        robot_num = random.randint(1, character_num)
        # allow robot num to different robot type
        
        random.shuffle(robot_types)
        # for robot_type in robot_types:
        #     if robot_num <= 0:
        #         config["met"]["robot"][f"{robot_type}_num"] = 0
        #     config["met"]["robot"][f"{robot_type}_num"] = random.randint(0, robot_num)
        #     robot_num -= config["met"]["robot"][f"{robot_type}_num"]
        for robot_type in robot_types:
            config["met"]["robot"][f"{robot_type}_num"] = 0

        for _ in range(robot_num):
            robot_type = random.choice(robot_types)
            config["met"]["robot"][f"{robot_type}_num"] += 1
            

        output_dir = f"/isaac-sim/tmp/mettest/data/{scene_path.split('/')[-1].split('.')[0]}_{i+1:03}"
        config["met"]["replicator"]["parameters"]["output_dir"] = output_dir
        config["met"]["character"]["command_file"] = f"{output_dir}/character_command.txt"
        config["met"]["robot"]["command_file"] = f"{output_dir}/robot_command.txt"

        yaml.dump(config, open(f"{config_save_dir}/{scene_path.split('/')[-1].split('.')[0]}_{i+1:03}.yaml", "w"))
