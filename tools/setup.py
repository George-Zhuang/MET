# find the isaacsim package path
import os
import json
import isaacsim

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /home/user/codes/met
isaacsim_path = os.path.dirname(isaacsim.__file__) # /home/user/anaconda3/envs/met/lib/python3.10/site-packages/isaacsim

# create softlink of exts/metspace.core to isaacsim/extsUser
print('*'*100)
print("[INFO] Setting up metspace.core for IsaacSim")
metspacecore_path = os.path.join(project_root, "exts/metspace.core")
isaacsim_extsuser_path = os.path.join(isaacsim_path, "extsUser")
if not os.path.exists(isaacsim_extsuser_path):
    os.system(f"ln -s {metspacecore_path} {isaacsim_extsuser_path}")
    print(f"Created softlink {isaacsim_extsuser_path} to {project_root}/exts/metspace.core")
else:
    print(f"{isaacsim_extsuser_path} already exists, skipping... If you want to update, please remove it manually.")

# create softlink of isaacsim to app
print('*'*100)
print("[INFO] Setting up app for IsaacSim")
app_path = os.path.join(project_root, "app")
if not os.path.exists(app_path):
    # create softlink of isaacsim core to app
    os.system(f"ln -s {isaacsim_path} {app_path}")
    print(f"Created softlink {app_path} to {isaacsim_path}")
else:
    print(f"{app_path} already exists, skipping... If you want to update, please remove it manually.")

# create vscode settings
print('*'*100)
print("[INFO] Setting up vscode settings for IsaacSim")
vscode_dir = os.path.join(project_root, ".vscode")
if not os.path.exists(vscode_dir):
    os.makedirs(vscode_dir)
    print(f"Created {vscode_dir}")
vscode_settings = os.path.join(vscode_dir, "settings.json")
with open(vscode_settings, "w") as f:
    # write settings
    settings = {"python.analysis.extraPaths": []}
    # walk through exts, extscache, extsPhysics
    exts_path = os.path.join(app_path, "exts")
    extscache_path = os.path.join(app_path, "extscache")
    extsPhysics_path = os.path.join(app_path, "extsPhysics")
    # append all the dirs in exts_path with depth 1
    settings["python.analysis.extraPaths"].append(metspacecore_path)
    settings["python.analysis.extraPaths"] += [os.path.join(exts_path, d) for d in os.listdir(exts_path) if os.path.isdir(os.path.join(exts_path, d))]
    settings["python.analysis.extraPaths"] += [os.path.join(extscache_path, d) for d in os.listdir(extscache_path) if os.path.isdir(os.path.join(extscache_path, d))]
    settings["python.analysis.extraPaths"] += [os.path.join(extsPhysics_path, d) for d in os.listdir(extsPhysics_path) if os.path.isdir(os.path.join(extsPhysics_path, d))]
    json.dump(settings, f, indent=4)
print(f"Created {vscode_settings}.")

print('*'*100)
print(f"[WARN] The added extensions do not cover all extensions, but they can meet basic needs.")
