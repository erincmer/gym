import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from blochsimu import Simulator
import random
from scipy.integrate import simps
from itertools import product

class Continuous_MountainCarEnv(gym.Env):

    def _B0_field(self,x, y, z):
        return np.array([0, 0, 0.0001*( -1*((x-0.5*self.x_len)/self.x_len) + 0.6*((x-0.5*self.x_len)/self.x_len)*((x-0.5*self.x_len)/self.x_len) - 3*((x-0.5*self.x_len)/self.x_len)*((x-0.5*self.x_len)/self.x_len)*((x-0.5*self.x_len)/self.x_len))]) #10 ppm

    def _B1_field(self,x, y, z):
        return np.array([0, 0, 1])

    def _Gx_field(self,x, y, z):
        return np.array([0, 0, ((x-0.5*self.x_len)/self.x_len)])

    def __init__(self,act_size = 3, size_x= 21,size_y = 1,size_z = 1, init_state=10., state_bound=1):

        self.x_len, self.y_len, self.z_len = size_x, size_y, size_z

        self.size = self.x_len * self.y_len * self.z_len * 3

        self.init_state = np.zeros((self.size))

        self.init_state[::3]=1

        self.action_space = spaces.Box(low=-state_bound, high=state_bound, shape=(act_size,))

        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(self.size,))

        self._seed()

        self.done = False

        self.pi = 3.1415

        self.time_val = 1.65*pow(10, -3)              # Experiment length

        self.maxB1=0.002

        self.time_limit = 50

        self.time_counter = 0

        self.dt_val = self.time_val/self.time_limit              # Time between plays

        self.numerical_steps= int(10*self.dt_val*(self.maxB1*42*1000000))               # Number of differential steps after a play

        self.B0 = np.zeros([self.x_len, self.y_len, self.z_len, 3])

        self.B1 = np.zeros([self.x_len, self.y_len, self.z_len,3])

        self.grad_X = np.zeros([self.x_len, self.y_len, self.z_len,3])

        for i,j,k in product(range(self.x_len), range(self.y_len), range(self.z_len)):
            self.B0[i, j, k, :] = self._B0_field(i, j, k)
            self.B1[i, j, k, :] = self._B1_field(i, j, k)
            self.grad_X[i, j, k, :] = self._Gx_field(i, j, k)
           

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _single_spin_stepper(self,single_state, field):
        sim = Simulator(single_state, n_steps = self.numerical_steps)
        for dt in range(self.numerical_steps):
          single_state = sim.simulate_step(field, self.dt_val/self.numerical_steps)
        single_state=np.array(single_state)/np.linalg.norm(single_state)
        return single_state

    def _step(self,u):

        reward = self._reward(u)
        ind = 0

        for i,j,k in product(range(self.x_len), range(self.y_len), range(self.z_len)):
            point_field_1 = self.B0[i, j, k] + u[2]*self.grad_X[i, j, k]*self.maxB1/10
            point_field_4 = [u[0]*math.sin(u[1]*self.pi)*self.maxB1,u[0]*math.cos(u[1]*self.pi)*self.maxB1,0]
            point_field = point_field_1 + point_field_4
            self.state[ind:ind+3] = self._single_spin_stepper(self.state[ind:ind+3] , point_field)
            ind+=3

        if self.time_counter == self.time_limit:
            self.done = True
            return self._reset(), reward, True, {}
        self.time_counter +=1
        return self._get_obs(), reward, self.done, {}


    def _reward(self,u):
          return np.abs(np.sum(self.state[::3])/(self.x_len*self.y_len*self.z_len))

    def _reset(self):
        self.state = self.init_state
        self.done = False
        self.time_counter = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state
