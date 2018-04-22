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
        self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        self.last_rotor_speeds = []
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 10.

        reward -= abs(self.sim.v[:2]).sum() # x, y velocities must be 0
        reward -= abs(self.sim.v[2]) # penalize z velocity
        reward -= abs(self.sim.angular_v).sum() # angular velocities must be 0
        reward -= abs(self.sim.pose[3:]).sum() # euler angles must be 0

        reward -= abs(self.sim.linear_accel[2])

        reward -= abs(self.target_pos[2] - self.sim.pose[2]) * 2

        # reward -= np.var(self.last_rotor_speeds)

        # print("V = {}".format(self.sim.v[2]))
        # print("R = {}".format(abs(self.sim.v[2]) * factor))
        # print("H = {}".format(self.sim.pose[2]))

        # distance = np.linalg.norm(abs(self.sim.pose[:3] - self.target_pos))
        # reward = np.tanh(1 - 0.006*(distance))
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
