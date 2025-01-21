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

import pdb
import math
import random

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.env_targets = np.zeros((cfg.num_rows, cfg.num_cols, 2))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)+500
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.25, 0.375, 0.45])
            # difficulty = 0
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                # difficulty = 0
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.3
        step_height = 0.0 + 0.1 * difficulty
        target_height = 0.0 + 3.0 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.075
        stepping_stones_size = 0.75 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.05
        gap_size = 0.35 * difficulty
        pit_depth = 0.3 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.5)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=1.5)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.75, step_height=step_height, platform_size=2)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=1.5)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            new_stair_terrain(terrain, step_width=1, step_height=step_height)
        elif choice < self.proportions[8]:
            new_stair_terrain(terrain, step_width=1, step_height=-step_height)
        elif choice < self.proportions[9]: 
            new_slope_terrain(terrain, step_width=1, target_height=target_height)
        elif choice < self.proportions[10]: 
            new_slope_terrain(terrain, step_width=1, target_height=-target_height)
        elif choice < self.proportions[11]: 
            new_rough_slope_terrain(terrain, step_width=1, target_height=target_height)
        elif choice < self.proportions[12]: 
            new_rough_slope_terrain(terrain, step_width=1, target_height=-target_height)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = (i) * self.env_length + 0.75
        env_origin_y = (j + 0.5) * self.env_width
        # x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        # x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        # y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        # y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        x1 = int((0) / terrain.horizontal_scale)
        x2 = int((0.3) / terrain.horizontal_scale)
        y1 = int((0.05) / terrain.horizontal_scale)
        y2 = int((self.env_width - 0.05) / terrain.horizontal_scale)
        # pdb.set_trace()
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        
        env_target_x = i * 10 + 9.5
        env_target_y = j * 10 + 5

        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.env_targets[i, j] = [env_target_x, env_target_y]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

def new_stair_terrain(terrain, step_width, step_height):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    height = 0
    
    start_width = 10
    end_width = 10

    start_x = start_width
    stop_x = start_width + step_width
    start_y = 0
    stop_y = terrain.width
    # print("Length: " +  str(terrain.length))
    # print("Width: " + str(terrain.width))
    # pdb.set_trace()
    while (terrain.length - stop_x) > end_width:
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
        start_x = stop_x
        stop_x += step_width
    terrain.height_field_raw[start_x: terrain.width, start_y: stop_y] = height
    return terrain

def new_slope_terrain(terrain, step_width, target_height):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = 0.1
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(target_height / 80 / terrain.vertical_scale)

    height = 0
    
    start_width = 10
    end_width = 10

    start_x = start_width
    stop_x = start_width + step_width
    start_y = 0
    stop_y = terrain.width
    # print("Length: " +  str(terrain.length))
    # print("Width: " + str(terrain.width))
    # pdb.set_trace()
    while (terrain.length - stop_x) > end_width:
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
        start_x = stop_x
        stop_x += step_width
    terrain.height_field_raw[start_x: terrain.width, start_y: stop_y] = height
    return terrain
def new_rough_slope_terrain(terrain, step_width, target_height):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = 0.1
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(target_height / 80 / terrain.vertical_scale)

    height = 0
    
    start_width = 10
    end_width = 10

    start_x = start_width
    stop_x = start_width + step_width
    start_y = 0
    stop_y = terrain.width
    # print("Length: " +  str(terrain.length))
    # print("Width: " + str(terrain.width))
    # pdb.set_trace()
    while (terrain.length - stop_x) > end_width:
        height += step_height
        i = 0
        while i < terrain.width:
            terrain.height_field_raw[start_x: stop_x, i: i+1] = height + random.random()*(0.05/terrain.vertical_scale)
            i = i+1
        start_x = stop_x
        stop_x += step_width
    terrain.height_field_raw[start_x: terrain.width, start_y: stop_y] = height
    return terrain