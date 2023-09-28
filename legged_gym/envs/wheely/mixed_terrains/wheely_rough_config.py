# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from isaacgym import torch_utils as tu
import torch as t
import numpy as np

class WheelyRoughCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 16
        max_iterations=500


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.25] # x,y,z [m]
        rot = [0, 0, 0, 1]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "FLHU": 0.0,
            "BLHU": 0.0,
            "FRHU": 0.0,
            "BRHU": 0.0,

            "FLHL": 0,
            "BLHL": 0,
            "FRHL": 0,
            "BRHL": 0,

            "FLK": 0,
            "BLK": 0,
            "FRK": 0,
            "BRK": 0,

            # "FLHU": np.deg2rad(45),
            # "BLHU": np.deg2rad(45),
            # "FRHU": -np.deg2rad(45),
            # "BRHU": -np.deg2rad(45),

            "FLHL": np.deg2rad(45),
            "BLHL": np.deg2rad(45),
            "FRHL": -np.deg2rad(45),
            "BRHL": -np.deg2rad(45),

            "FLK": -np.deg2rad(90),
            "BLK": -np.deg2rad(90),
            "FRK": np.deg2rad(90),
            "BRK": np.deg2rad(90),

            "FLW": 0,
            "BLW": 0,
            "FRW": 0,
            "BRW": 0
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'HU': 20, 'HL': 20, 'K': 20, 'W': 20}  # [N*m/rad]
        damping = {'HU': 0.5, 'HL': 0.5, 'K': 0.5, 'W': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wheely/URDFs/V2-8.urdf"
        name = "Wheely"
        disable_gravity = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.175
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.80 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 10 # forces above this value are penalized
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -4
            tracking_lin_vel = 10#-0.25
            tracking_ang_vel = 5#-0.25
            lin_vel_z = -2#-0.25
            ang_vel_xy = -0.05#-0.25
            orientation = -1.5
            torques = 0#-0.00001
            dof_vel = 0#-0.00001
            dof_acc = 0#-0.00001
            base_height = -2.5
            feet_air_time = 1.5 #0.0
            collision = -1
            base_collision = -1.25
            feet_collision = 0 #.1
            feet_stumble = -0#-0.0 
            action_rate = 0#-0.01
            stand_still = 0#-0.
            episode_length=0.001 #logarithmically increasing reward
            # terrain=1 

class WheelyRoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_wheely'
        load_run = -1
