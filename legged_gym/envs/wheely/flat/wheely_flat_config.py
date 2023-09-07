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
import numpy as np
from legged_gym.envs import WheelyRoughCfg, WheelyRoughCfgPPO

class WheelyFlatCfg( WheelyRoughCfg ):
    class env( WheelyRoughCfg.env ):
        num_observations = 60
  
    class terrain( WheelyRoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset(WheelyRoughCfg.asset ):
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False

    class rewards( WheelyRoughCfg.rewards ):
        base_height_target = 0.25
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.90 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 25 # forces above this value are penalized
        class scales( WheelyRoughCfg.rewards.scales ):
            termination = -3
            tracking_lin_vel = 2.5#-0.25
            tracking_ang_vel = 1.25#-0.25
            lin_vel_z = -0.5#-0.25
            ang_vel_xy = -0.05#-0.25
            orientation = -1
            torques = 0#-0.00001
            dof_vel = 0#-0.00001
            dof_acc = 0#-0.00001
            base_height = -2.5
            feet_air_time = 0.75 #0.0
            collision = -1
            base_collision = 0
            feet_collision = 0 #.1
            feet_stumble = -0#-0.0 
            action_rate = 0#-0.01
            stand_still = 0#-0.
            episode_length=0.001 #logarithmically increasing reward
            # terrain=1 
    class init_state( WheelyRoughCfg.init_state ):
        pos = [0.0, 0.0, 0.2] # x,y,z [m]
        rot = [0, 0, 0, 1]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "FLHU joint": 0.0,
            "BLHU joint": 0.0,
            "FRHU joint": 0.0,
            "BRHU joint": 0.0,

            "FLHL joint": 0,
            "BLHL joint": 0,
            "FRHL joint": 0,
            "BRHL joint": 0,

            # "FLK joint": 0,
            # "BLK joint": 0,
            # "FRK joint": 0,
            # "BRK joint": 0,

            # "FLHU joint": np.deg2rad(45),
            # "BLHU joint": np.deg2rad(45),
            # "FRHU joint": -np.deg2rad(45),
            # "BRHU joint": -np.deg2rad(45),

            # "FLHL joint": np.deg2rad(15),
            # "BLHL joint": np.deg2rad(15),
            # "FRHL joint": -np.deg2rad(15),
            # "BRHL joint": -np.deg2rad(15),

            "FLK joint": -np.deg2rad(60),
            "BLK joint": -np.deg2rad(60),
            "FRK joint": np.deg2rad(60),
            "BRK joint": np.deg2rad(60),

            "FLW joint": 0,
            "BLW joint": 0,
            "FRW joint": 0,
            "BRW joint": 0
        }
    class domain_rand( WheelyRoughCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

class WheelyFlatCfgPPO( WheelyRoughCfgPPO ):
    class policy( WheelyRoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( WheelyRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( WheelyRoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_wheely'
        load_run = -1
        max_iterations = 3000
