# GraspGenX uv Inference Setup

This guide sets up GraspGenX for inference with `uv`, including the model
checkpoints and gripper assets.

## 1. Install System Prerequisites

Install Git LFS before downloading checkpoint assets:

```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone And Install

Clone the Talos fork:

```bash
cd ~/talos-dev
git clone https://github.com/talos-robotics-ai/GraspGenX.git
cd GraspGenX
uv sync
```

For inference, the `uv` install is enough. Docker is mainly recommended for
training.

## 3. Download Checkpoints And Gripper Assets

GraspGenX auto-clones these repos on first import:

- `ext/graspgenx_checkpoints`
- `ext/gripper_descriptions`

Trigger that setup:

```bash
uv run python -c "from graspgenx import get_checkpoints_version_dir, get_gripper_descriptions_root; print(get_checkpoints_version_dir()); print(get_gripper_descriptions_root())"
```

Then pull the real Git LFS files:

```bash
cd ~/talos-dev/GraspGenX/ext/graspgenx_checkpoints
git lfs pull

cd ~/talos-dev/GraspGenX/ext/gripper_descriptions
git lfs pull
```

Verify that checkpoint files are real files, not tiny LFS pointers:

```bash
cd ~/talos-dev/GraspGenX/ext/graspgenx_checkpoints
wc -c release/gen/config.yaml release/gen/epoch_736.pth release/dis/config.yaml release/dis/epoch_1056.pth
```

Expected approximate sizes:

```text
release/gen/config.yaml       ~4 KB
release/gen/epoch_736.pth     ~1.21 GB
release/dis/config.yaml       ~4 KB
release/dis/epoch_1056.pth    ~0.48 GB
```

Total checkpoint download is about 1.7 GB. Gripper assets add roughly another
few hundred MB.

## 4. Verify Gripper Assets

List supported grippers:

```bash
cd ~/talos-dev/GraspGenX
uv run python scripts/list_grippers.py
```

Common supported names include:

```text
franka_panda
robotiq_2f_85
robotiq_2f_140
unitree_g1
inspire_hand
surge_hand
barrett_hand
galaxea_g1
ezgripper
```

## 5. Run Point Cloud Inference Demo

```bash
cd ~/talos-dev/GraspGenX
uv run python scripts/demo_object_pc.py \
  --sample_data_dir assets/sample_data/real_world \
  --gripper_name robotiq_2f_85 \
  --plot_top_mesh
```

Open the Viser URL printed in the terminal, usually:

```text
http://localhost:8080
```

## 6. Run Mesh Inference Demo

```bash
cd ~/talos-dev/GraspGenX
uv run python scripts/demo_object_mesh.py \
  --mesh_file assets/sample_data/object_mesh/banana.obj \
  --mesh_scale 1.0 \
  --gripper_name inspire_hand \
  --grasp_threshold -1.0 \
  --return_topk \
  --topk_num_grasps 100 \
  --plot_top_mesh
```

To save grasps to YAML:

```bash
uv run python scripts/demo_object_mesh.py \
  --mesh_file assets/sample_data/object_mesh/banana.obj \
  --mesh_scale 1.0 \
  --gripper_name inspire_hand \
  --grasp_threshold -1.0 \
  --return_topk \
  --topk_num_grasps 100 \
  --plot_top_mesh \
  --output_file /tmp/graspgenx_grasps.yml
```

## 7. Optional External Asset Locations

If checkpoints or gripper descriptions are stored elsewhere:

```bash
export GRASPGENX_CHECKPOINT_DIR=/path/to/graspgenx_checkpoints
export GRASPGENX_GRIPPER_CFG_DIR=/path/to/gripper_descriptions
```

Then rerun the demo command.

## Troubleshooting

If you see:

```text
omegaconf.errors.ConfigAttributeError: Missing key diffusion
```

then the checkpoints are still Git LFS pointer files. Run:

```bash
cd ~/talos-dev/GraspGenX/ext/graspgenx_checkpoints
git lfs pull
```

If `git lfs` is not recognized:

```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
```
