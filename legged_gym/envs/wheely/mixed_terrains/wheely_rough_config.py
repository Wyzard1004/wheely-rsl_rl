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
        # num_observations = 48 # Not Wheeled
        num_observations = 60 # Wheeled
        num_actions = 16 #Wheeled
        # num_actions = 12 #Not Wheeled


    class terrain(LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.375] # x,y,z [m]
        rot = [0.0, 0.0, 0.707, 0.707]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # "FLHU": 0.0,
            # "BLHU": 0.0,
            # "FRHU": 0.0,
            # "BRHU": 0.0,

            # "FLHL": 0,
            # "BLHL": 0,
            # "FRHL": 0,
            # "BRHL": 0,

            # "FLK": 0,
            # "BLK": 0,
            # "FRK": 0,
            # "BRK": 0,

            "FLHU": -np.deg2rad(5),
            "BLHU": np.deg2rad(5),
            "FRHU": np.deg2rad(5),
            "BRHU": -np.deg2rad(5),

            "FLHL": np.deg2rad(30),
            "BLHL": np.deg2rad(30),
            "FRHL": -np.deg2rad(30),
            "BRHL": -np.deg2rad(30),

            "FLK": -np.deg2rad(60),
            "BLK": -np.deg2rad(60),
            "FRK": np.deg2rad(60),
            "BRK": np.deg2rad(60),


            # "FLHU": np.deg2rad(45),
            # "BLHU": np.deg2rad(45),
            # "FRHU": -np.deg2rad(45),
            # "BRHU": -np.deg2rad(45),

            # "FLHL": np.pi/4,
            # "BLHL": np.pi/4,
            # "FRHL": -np.pi/4,
            # "BRHL": -np.pi/4,

            # "FLK": -np.pi/2,
            # "BLK": -np.pi/2,
            # "FRK": np.pi/2,
            # "BRK": np.pi/2

            "FLW": 0,   
            "BLW": 0,
            "FRW": 0,
            "BRW": 0
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'': 20}  # [N*m/rad]
        damping = {'': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/wheely/URDFs/V2-9.urdf"
        name = "Wheely"
        disable_gravity = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.25
        only_positive_rewards = False
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1. 
        soft_torque_limit = 1. 
        max_contact_force = 50 # forces above this value are penalized
        class scales(LeggedRobotCfg.rewards.scales):
            # termination = -0
            # tracking_lin_vel = 5.0
            # tracking_ang_vel = 5.0
            # lin_vel_z = -2
            # ang_vel_xy = -0.005
            # orientation = -0.00
            # torques = -2.5e-7
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.00
            # feet_air_time = 0
            
            # feet_stumble = -0.0
            # action_rate_derivative = -0.0025
            # action_rate = -0.005  # this stops the agent from learning at the begining 
            # stand_still = -0.
            # # base_height = -0.01

            # collision = -2.5
            # base_collision = -5
            # feet_collision = 0.001 #
            # dof_pos_limits = -10
            # episode_length=0.000 #logarithmically increasing reward
            # terrain=0


            termination = -200
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            lin_vel_z = -0.00
            ang_vel_xy = -0.00
            orientation = -0.000
            torques = -2.5e-7
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.00
            feet_air_time = 0.1
            proximity = 0.1
            position = 10
            heading = 1

            
            feet_stumble = -0.0
            action_rate_derivative = -0.0025
            action_rate = -0.0001  # this stops the agent from learning at the begining 
            stand_still = -0.
            # base_height = -0.01

            collision = -2.5
            base_collision = -50
            feet_collision = 0.00 #
            dof_pos_limits = -10
            episode_length=0 #logarithmically increasing reward
            terrain=0



class WheelyRoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_wheely'
