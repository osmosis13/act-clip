# ACT-CLIP: Language-Conditioned Action Chunking Transformer with CLIP

#### This repository extends the Action Chunking Transformer (ACT) developed by Zhao et al. (https://tonyzhaozh.github.io/aloha/), by integrating CLIP ViT-B/32 as a visual-language backbone, enabling bimanual robotic manipulation conditioned on natural language instructions. The system is evaluated in the MuJoCo ALOHA simulation environment on a colour-conditioned cube transfer task, where the robot must transfer either a red or blue cube based on a natural language instruction.

### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` ACTPolicy adaptor with CLIP text encoding and dual language injection
- ``clip_encoder.py`` CLIPDualEncoder — frozen CLIP text and image patch encoder
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants and task configurations
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

### System Requirements

Tested on:
- Ubuntu 22.04
- Python 3.8.10
- CUDA-capable NVIDIA GPU
- PyTorch
- MuJoCo 2.3.7
- DM-Control 1.0.14

### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    pip install sentencepiece

## ACT Components 

    cd act-clip/detr
    pip install -e .

### Configure Dataset Directory

Edit ``constants.py`` to match your directory:

    DATA_DIR = "/path/to/datasets"

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Dataset Collection

We use ``sim_transfer_cube_color_scripted`` task in the examples below. To collect 200 demonstrations (101 red, 99 blue):

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_color_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 200

You can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube Color task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_color_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 3000  --lr 1e-5 \
    --seed 0

To enable temporal ensembling, add flag ``--temporal_agg``.
To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
During evaluation, the terminal will prompt for the instruction:

    ========================================
    INSTRUCTION QUERY
      1  →  pick up red cube
      2  →  pick up blue cube
      or type a custom instruction
    Enter choice:

The policy runs 50 rollouts per evaluation and reports:

    Success rate: 0.XX
    Reward >= 1: XX/50
    Reward >= 2: XX/50
    Reward >= 3: XX/50
    Reward >= 4: XX/50

Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.
Task parameters are defined in constants.py. Ensure the following entry exists:

    'sim_transfer_cube_color_scripted': {
    'dataset_dir': DATA_DIR + '/sim_transfer_cube_color_scripted',
    'num_episodes': 200,
    'episode_len': 400,
    'camera_names': ['angle', 'top']
    },

### CLIP Configurations

The CLIP encoder configuration is defined in:

``detr/models/detr_vae.py``

within the `build()` function.

Curently the configuration is set to unfreeze the last 2 transformer blocks of the ViT-B/32, with a reduced learning rate of 1e-6 instead of the current 1e-5 used in the erst fo the model. You can change the configuration to fully frozen ViT CLIP parameters by changing this parameter:

    clip_enc = CLIPDualEncoder(freeze=True, unfreeze_last_n_blocks=2)

and set ``unfreeze_last_n_blocks`` to ``0``.

### Acknowledgements

This repository builds upon the Action Chunking Transformer (ACT) framework developed by Zhao et al.:

https://tonyzhaozh.github.io/aloha/

CLIP was developed by OpenAI:

https://github.com/openai/CLIP