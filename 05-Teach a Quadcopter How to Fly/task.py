import numpy as np
from physics_sim import PhysicsSim
import math

def sigmoid(x):
    return 1. / (1. + math.exp(-x))

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.last_rotor_speeds = []
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.])


    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 0

        # reward -= np.linalg.norm(self.sim.pose[:3] - self.target_pos) * 0.3
        # reward -= np.linalg.norm(self.sim.v) * 0.09
        # reward -= np.linalg.norm(self.sim.linear_accel) * 0.027

        # reward -= (self.sim.find_body_velocity() ** 2).sum() # velocities must be 0
        # reward -= (self.sim.angular_v ** 2).sum() * .005# angular velocities must be 0
        # reward -= (self.sim.linear_accel ** 2).sum() * .005
        # reward -= (self.sim.angular_accels ** 2).sum() * .005


        # reward -= np.std(self.last_rotor_speeds)

        # distance = np.linalg.norm(abs(self.sim.pose[:3] - self.target_pos))
        # reward = np.tanh(1 - 0.006*(distance))

        # dist = np.linalg.norm(np.array(self.sim.pose[:3]) - np.array(self.target_pos))
        # reward = np.tanh(1.0 -.2*(dist))

        # distance_to_target = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        # acceleration = np.linalg.norm(self.sim.linear_accel)

        # reward = 0#np.linalg.norm(self.target_pos)
        # reward -= distance_to_target
        # reward -= acceleration
        # reward -= angles * 0.3

        # distance_z = np.linalg.norm(self.target_pos[2] - self.sim.pose[2])
        # distance_xy = np.linalg.norm(self.target_pos[:2] - self.sim.pose[:2])

        acceleration = np.linalg.norm(self.sim.linear_accel)
        # acceleration_z = np.linalg.norm(self.sim.linear_accel[2])
        # acceleration_xy = np.linalg.norm(self.sim.linear_accel[:2])

        # # speed = np.linalg.norm(self.sim.v)
        speed = np.linalg.norm(self.sim.find_body_velocity())
        distance = np.linalg.norm(self.target_pos - self.sim.pose[:3])
        # speed_z = np.linalg.norm(self.sim.v[2])

        # reward = - (np.linalg.norm(self.target_pos[2]) - distance_z) / np.linalg.norm(self.target_pos[2]) - acceleration_z / 25.
        # reward1 = 1. - 2. * math.tanh(speed_z)
        # reward2 = 1. - 2. * math.tanh(speed)
        # reward = reward1 * 0.7 + reward2 * 0.3
        MAX_SPEED = 28.
        MAX_ACCELERATION = 39.
        # reward = 1. - 2. * math.tanh(((speed / MAX_SPEED) + (acceleration / MAX_ACCELERATION)) * 2)
        speed_reward = speed / MAX_SPEED
        distance_reward = distance / np.linalg.norm(self.target_pos)
        acceleration_reward = acceleration / MAX_ACCELERATION

        reward = 1. - 2. * math.tanh(speed_reward + distance_reward + acceleration_reward)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        self.last_rotor_speeds = rotor_speeds
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) or self.sim.pose[2] <= 0.
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
