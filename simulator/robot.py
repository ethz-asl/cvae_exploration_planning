import numpy as np
import random
import math
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Wedge, Circle
from matplotlib.colors import ListedColormap
from cvae_exploration_planning.simulator.config import Config


class Robot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.x_buff = [0., 0., 0.]
        self.y_buff = [0., 0., 0.]
        self.yaw_buff = [0., 0., 0.]
        self.position_x = None  # in voxels and image coordinates
        self.position_y = None
        self.yaw = 0  # in [0, 2pi]
        self.observed_map = None  # 0 = free, 1 = occupied, 2 = unobserved
        self.world = None  # reference to the world the robot is in, not for planning use
        self.pos_ramp = VelocityRamp(
            cfg.velocity_max / cfg.voxel_size, cfg.acceleration / cfg.voxel_size)
        self.yaw_ramp = VelocityRamp(cfg.yaw_rate_max, cfg.yaw_acceleration)

        self.init_display = False
        self.file_name = 20

        if self.cfg.save_images:
            matplotlib.use('Agg')
    
    def load_initial_position(self, position):
        ''' load the presaved initial robot position for reproductive results'''
        self.position_x = position[0]
        self.position_y = position[1]
        self.yaw = position[2]
        self.observed_map = np.ones(np.shape(self.world.map_gt)) * 2
        self.take_sensor_measurement()
        self.observed_map[self.position_x, self.position_y] = 0
        if self.cfg.initial_margin_observed:
            tmp_fov = self.cfg.camera_fov
            tmp_range = self.cfg.camera_range
            self.cfg.camera_fov = 360
            self.cfg.camera_range = self.cfg.initial_safety_margin
            self.take_sensor_measurement()
            self.cfg.camera_fov = tmp_fov
            self.cfg.camera_range = tmp_range

    def initialize_in_world(self, world):
        """ Find an initial position in the world and reset stats"""
        self.world = world
        # Guarantee starting position meets criteria
        while True:
            self.position_x = random.randint(
                0, np.shape(self.world.map_gt)[0] - 1)
            self.position_y = random.randint(
                0, np.shape(self.world.map_gt)[1] - 1)
            if self.cfg.initial_safety_margin <= 0:
                if self.world.map_gt[self.position_x, self.position_y] == 0:
                    break
            else:
                offset = int(self.cfg.initial_safety_margin /
                             self.cfg.voxel_size)
                if self.position_x - offset < 0 or self.position_x + offset > world.map_size_x or \
                        self.position_y - offset < 0 or self.position_y + offset > world.map_size_y:
                    continue
                local_area = self.world.map_gt[self.position_x - offset:self.position_x + offset,
                                               self.position_y - offset:self.position_y + offset]
                if np.max(local_area) == 0:
                    # only free space in local square
                    break
        if self.cfg.initial_yaw is None:
            self.yaw = random.uniform(0, math.pi * 2)
        else:
            self.yaw = self.cfg.initial_yaw
        self.observed_map = np.ones(np.shape(self.world.map_gt)) * 2
        self.take_sensor_measurement()
        self.observed_map[self.position_x, self.position_y] = 0
        if self.cfg.initial_margin_observed:
            tmp_fov = self.cfg.camera_fov
            tmp_range = self.cfg.camera_range
            self.cfg.camera_fov = 360
            self.cfg.camera_range = self.cfg.initial_safety_margin
            self.take_sensor_measurement()
            self.cfg.camera_fov = tmp_fov
            self.cfg.camera_range = tmp_range

    def bound_query_voxels(self, x, y):
        """ Floor and crop x and y to map size """
        x_out = np.maximum(np.minimum(
            np.round(x), np.shape(self.world.map_gt)[0] - 1), 0)
        y_out = np.maximum(np.minimum(
            np.round(y), np.shape(self.world.map_gt)[1] - 1), 0)
        return np.array(x_out, dtype=int), np.array(y_out, dtype=int)

    def take_sensor_measurement(self):
        """ Does naive ray-casting to identify visible voxels """
        n_rays = self.cfg.camera_range * math.pi * 2 * \
            self.cfg.camera_fov / 360 / self.cfg.voxel_size
        # make sure we get full coverage of voxels,
        n_rays = math.ceil(n_rays * 1.5)
        dphi = self.cfg.camera_fov * math.pi / 180 / (n_rays - 1)
        for i in range(n_rays):
            r = 0
            phi = self.yaw - self.cfg.camera_fov * math.pi / 180 / 2 + i * dphi
            while r <= self.cfg.camera_range / self.cfg.voxel_size:
                r = r + 1
                x, y = self.bound_query_voxels(
                    self.position_x + r * math.cos(phi), self.position_y + r * math.sin(phi))
                self.observed_map[x, y] = self.world.map_gt[x, y]
                if self.observed_map[x, y] == 1:
                    break

    def check_move_feasible(self, x, y, step=None, vect=False, x_start=0, y_start=0):
        """ Check whether move collides and is safe during planning. Coordinates are in robot-centric frame. """
        if step is None:
            step = self.cfg.voxel_size / 2  # double resolution to catch all voxels
        n_steps = math.ceil(
            ((x-x_start) ** 2 + (y - y_start) ** 2) ** 0.5 / step)
        x_vals = self.position_x + np.linspace(x_start, x, max(1, n_steps))
        y_vals = self.position_y + np.linspace(y_start, y, max(1, n_steps))
        if vect:
            px, py = self.bound_query_voxels(x_vals, y_vals)
            obs_map = self.observed_map[px, py] != 0
            coll_map = self.world.map_gt[px, py] != 0
            is_safe = sum(obs_map) == 0
            coll_step = np.argmax(coll_map)
            try:
                has_collided_at = float(
                    coll_step) / n_steps if (coll_step > 0) or (coll_step == 0 and coll_map[0]) else None
            except:
                print(f"Collision check failed!")
                print(f"x={x}, y={y}, n_steps={n_steps},\nx_vals={x_vals}, y_vals={y_vals},\n"
                      f"px={px}, py={py}, \nobs_map={obs_map}, coll_map={coll_map},\n"
                      f"is_safe={is_safe}, coll_step={coll_step}")
                has_collided_at = None
        else:
            is_safe, has_collided_at = True, None
            for i in range(n_steps):
                px, py = self.bound_query_voxels(self.position_x + x * (i + 1) / n_steps,
                                                 self.position_y + y * (i + 1) / n_steps)
                if self.observed_map[px, py] != 0:
                    is_safe = False
                if self.world.map_gt[px, py] != 0:
                    has_collided_at = i / n_steps
                    break
        return has_collided_at, is_safe

    def move_to(self, x, y, yaw):
        """ move to point relative to current pose and to specified yaw. Uses velocity ramp model for path and yaw
        separately and sample the viewpoints. Collision is not checked here! """
        distance = (float(x) ** 2 + float(y) ** 2) ** 0.5
        # Find shortest yaw distance
        yaw_diff = math.fmod(yaw - self.yaw, math.pi * 2)
        if math.fabs(yaw_diff) > math.pi:
            yaw_diff = yaw_diff - np.sign(yaw_diff) * math.pi * 2
        self.pos_ramp.setup_from_goal(distance)
        # requires positive sign, so compensate for it later
        self.yaw_ramp.setup_from_goal(math.fabs(yaw_diff))
        max_time = max(self.pos_ramp.t_max, self.yaw_ramp.t_max)
        self.x_buff[0], self.y_buff[0], self.yaw_buff[0] = self.position_x, self.position_y, self.yaw
        self.x_buff[2], self.y_buff[2], self.yaw_buff[2] = x + \
            self.x_buff[0], y + self.y_buff[0], yaw
        for i in range(math.ceil(max_time * self.cfg.camera_rate) + 1):
            # step along the path and take measurements
            t = i / self.cfg.camera_rate
            d = self.pos_ramp.x(t)
            if distance > 0:
                self.position_x = self.x_buff[0] + x * d / distance
                self.position_y = self.y_buff[0] + y * d / distance
                self.x_buff[1] = self.position_x
                self.y_buff[1] = self.position_y
            self.yaw = self.yaw_buff[0] + \
                np.sign(yaw_diff) * self.yaw_ramp.x(t)
            self.x_buff[1], self.y_buff[1], self.yaw_buff[1] = self.position_x, self.position_y, self.yaw
            self.take_sensor_measurement()
            if self.cfg.save_images:
                self.save_images()
        self.yaw = math.fmod(self.yaw, 2 * math.pi)
        if self.yaw < 0:
            self.yaw = self.yaw + math.pi * 2
        return distance, max_time

    def save_images(self, action=None, collision=False):
        # First time, create objects
        if action is None:
            action = [0, 0, 0]
        if not self.init_display:
            free = np.array([1, 1, 1, 1])
            occ = np.array([0, 0, 0, 1])
            unknown = np.array([0.7, 0.7, 0.7, 1])

            robot_cmp = ListedColormap(np.vstack((free, occ, unknown)))
            robot_map = self.observed_map
            self.fig = plt.figure(figsize=(20, 20))
            self.plot_ax = self.fig.subplots()
            self.image = plt.imshow(
                robot_map, cmap=robot_cmp, interpolation='none')
            self.line1, = self.plot_ax.plot(
                0., 0., linestyle=':', color='orange')
            self.plot_ax.fill_between(
                [.0, .1], [.0, .1], color='k', alpha=0.25)
            plt.axis([0, np.shape(self.world.map_gt)[0] - 1,
                     0, np.shape(self.world.map_gt)[1] - 1])
            plt.axis('off')
            self.fig.savefig(self.cfg.images_folder + str(self.file_name).zfill(8) +
                             '.png', bbox_inches='tight', pad_inches=0)
            self.file_name += 1
            self.init_display = True
            return

        new_color = 'red'
        #  If objects are created, update them every step
        x_new, y_new, yaw_new = self.position_x, self.position_y, self.yaw
        max_lim = self.cfg.world_size_x / self.cfg.voxel_size - 1
        x_lim = np.clip(x_new + self.cfg.local_submap_size_x /
                        2 * np.array([[-1, -1], [+1, +1]]), 0., max_lim)
        y_lim = np.clip(y_new + self.cfg.local_submap_size_y /
                        2 * np.array([-1, +1]), 0., max_lim)
        robot_map = self.observed_map
        self.image.set_data(robot_map)
        if collision:
            self.x_buff[1] = self.x_buff[0] + action[0]
            self.y_buff[1] = self.y_buff[0] + action[1]
            self.yaw_buff[1] = action[2]
        else:
            new_color = 'green'
        self.plot_ax.collections.clear()
        self.plot_ax.fill_between(
            y_lim, x_lim[0], x_lim[1], facecolor='k', alpha=0.1, edgecolor='k')
        wedge1 = Wedge((self.y_buff[1], self.x_buff[1]), self.cfg.camera_range / self.cfg.voxel_size,
                       -self.yaw_buff[1] * 180 / 3.14 + 45, -self.yaw_buff[1] * 180 / 3.14 + 135, color='blue', alpha=0.1)
        circle1 = Circle((self.y_buff[1], self.x_buff[1]), radius=self.cfg.camera_range / self.cfg.voxel_size / 15,
                         color='blue', alpha=0.75)
        circle1_patch = self.plot_ax.add_patch(circle1)
        wedge1_patch = self.plot_ax.add_patch(wedge1)
        arrow0 = Arrow(self.y_buff[2], self.x_buff[2],
                       1.5 * math.sin(self.yaw_buff[2]) / self.cfg.voxel_size,
                       1.5 * math.cos(self.yaw_buff[2]) / self.cfg.voxel_size,
                       width=1.0 / self.cfg.voxel_size, color=new_color)
        arrow0_patch = self.plot_ax.add_patch(arrow0)
        # self.fig.canvas.draw()
        plt.axis([0, np.shape(self.world.map_gt)[0] - 1,
                 0, np.shape(self.world.map_gt)[1] - 1])
        plt.axis('off')
        self.fig.savefig(self.cfg.images_folder + str(self.file_name).zfill(8) +
                         '.png', bbox_inches='tight', pad_inches=0)
        self.file_name += 1
        wedge1_patch.remove()
        circle1_patch.remove()
        arrow0_patch.remove()

    def get_local_submap(self):
        x_start = math.floor(
            self.position_x - self.cfg.local_submap_size_x / 2)
        y_start = math.floor(
            self.position_y - self.cfg.local_submap_size_y / 2)
        x_end = x_start + self.cfg.local_submap_size_x
        y_end = y_start + self.cfg.local_submap_size_y
        x_local_start = 0
        y_local_start = 0
        x_local_end = self.cfg.local_submap_size_x
        y_local_end = self.cfg.local_submap_size_y
        x_max = np.shape(self.observed_map)[0]
        y_max = np.shape(self.observed_map)[1]
        if x_start < 0:
            x_local_start = -x_start
            x_start = 0
        if x_end > x_max:
            x_local_end = x_max - x_start
            x_end = x_max
        if y_start < 0:
            y_local_start = -y_start
            y_start = 0
        if y_end > y_max:
            y_local_end = y_max - y_start
            y_end = y_max
        local_submap = np.ones(
            (int(self.cfg.local_submap_size_x), int(self.cfg.local_submap_size_y))) * 2
        local_submap[int(x_local_start):int(x_local_end), int(y_local_start):int(y_local_end)] = \
            self.observed_map[int(x_start):int(x_end), int(y_start):int(y_end)]
        return local_submap


class VelocityRamp:
    """ Simple 1D velocity ramp model against time x(t), assuming it starts at x(0) = 0 and x_goal > 0 """

    def __init__(self, v_max, a_max):
        self.x_max = 0
        self.x1 = 0  # saturation points
        self.x2 = 0
        self.t_max = 0
        self.t1 = 0
        self.t2 = 0
        self.v = v_max
        self.a = a_max
        self.mode_saturated = False

    def setup_from_goal(self, x_goal):
        self.x_max = x_goal
        if x_goal > self.v ** 2 / self.a:
            self.mode_saturated = True
            self.t_max = self.v / self.a + x_goal / self.v
            self.t1 = self.v / self.a
            self.t2 = self.t_max - self.t1
            self.x1 = self.v ** 2 / 2 / self.a
            self.x2 = self.x_max - self.x1
        else:
            self.mode_saturated = False
            self.t1 = (x_goal / self.a) ** 0.5
            self.t_max = 2 * self.t1
            self.x1 = self.x_max / 2

    def x(self, t):
        """ Query x(t) """
        if t >= self.t_max:
            return self.x_max
        if t <= self.t1:
            return self.a / 2 * t ** 2
        if self.mode_saturated:
            if t <= self.t2:
                return self.x1 + self.v * (t - self.t1)
            else:
                return self.x2 + self.v * (t - self.t2) - self.a / 2 * (t - self.t2) ** 2
        else:
            return self.x1 + self.a * self.t1 * (t - self.t1) - self.a / 2 * (t - self.t1) ** 2
