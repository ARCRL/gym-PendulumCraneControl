"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy import signal
from gym.envs.classic_control import rendering

class CartPoleEnv_Crane(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 2.0
        self.masspole = 0.2
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.7 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Motor constant
        R = 1.34
        L = 220 * 10**(-5)
        k_e = 20.9*10**(-3)
        k_t = k_e
        N = 45
        J = 52 * 10**(-7)
        b = k_e*0.125
        r_wd = 0.03
        k_b = k_t

        k_r = k_t/R

        l = self.polemass_length*0.7

        r_1 = r_wd/N

        c1 = k_r*k_b/r_1**2;
        c2 = k_r/r_1;

        pend_damp = -0.99

        A_ss = np.array([[0,1,0,0],\
        [(-self.masscart+self.masspole)*self.gravity/(self.masscart*l), pend_damp,0,c1/(l*self.masscart)],\
        [0, 0, 0, 1], \
        [self.masspole/self.masscart*self.gravity, 0, 0, -c1/self.masscart]])
        B_ss = np.array([[0],[-c2/(l*self.masscart)], [0], [c2/(self.masscart)]])
        C_ss = np.eye(4)
        D_ss = np.array([[0], [0], [0], [0]])

        system = signal.StateSpace(A_ss, B_ss, C_ss, D_ss)
        self.system = system.to_discrete(self.tau)

        print(self.system)

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 360 * math.pi / 180
        self.theta_threshold_radians = 120 * math.pi / 180

        # Set threshholds so it won't move outside the physical track
        # UPDATE
        self.x_threshold = 1.0
        self.x_threshold_upper = 0.6
        self.x_threshold_lower = -0.6

        self.goal_x = -0.20

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Continuous(20)
        #self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


        self.seed()
        self.viewer = None
        self.state = None
        self.action = self.action_space.sample()
        self.reward = 0

        self.steps_beyond_done = None
        self.reset()

    def set_goal(self, goal=None):
    	if goal == None:
    		goal = (self.np_random.random_sample(1)[0] * 2 - 1) * self.x_threshold_upper
    	self.goal_x = np.clip(goal, self.x_threshold_lower, self.x_threshold_upper)
    	self.state[4] = self.goal_x
    	
    def get_goal(self):
    	return self.goal_x

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_calc(self, distance, theta_dot, theta, x_dot, x, goal = None):
        #print(_dot)
        #print("{:3.4f} {:3.4f}".format(x_dot, theta_dot))
        #if x_dot == 0:
        #    penalty = 0
        #else:
        if goal == None:
            goal = self.goal_x
        sigma = 0.1
        mu = goal
        if goal == x:
            sign = 1
        else:
            sign = (x-goal)/abs(x-goal)
        distance = np.exp(-sign*(x-goal)*10*abs(x_dot))#1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))#np.exp(-(self.goal_x - x)) #max(abs(theta*10), abs(theta_dot), abs(x_dot*10), np.exp(-max(abs(theta*10), abs(theta_dot), abs(x_dot*10))) 
        #sigma = 0.1
        #mu = 0
        #velocity = -1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x_dot-mu)**2/(2*sigma**2))
        #ang_vel = 1-np.exp(-(theta_dot)/abs(theta_dot)*(theta_dot))
        reward = distance #+ velocity
        distance = abs(x-goal)
        boundary_distance = self.x_threshold_upper - abs(x)
        
        if boundary_distance < 0.05:
            reward = -1
        elif distance > 0.05:
            reward = 0
        else:
            reward = 1

        a = 10
        reward = np.exp(-a**2/2*(x-goal)**2) 
        return reward

    def get_reward(self):
        return self.reward

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        self.action = action
        x, x_dot, theta, theta_dot, goal = state

        V_a = action
        """
        force = self.force_mag if action==1 else -self.force_mag

        force = 0

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        """
        xk = np.matmul(self.system.A, np.array([theta, theta_dot, x, x_dot])) + self.system.B.T * V_a

        xk = xk[0]

        # print(xk)


        # Extract state variables
        theta = xk[0]
        theta_dot = xk[1]
        x = xk[2]
        x_dot = xk[3]
        # Stay witin the physical rig
        """
        if x > 2:
            x = 2
            x_dot = 0
        if x < 0:
            x = 0
            x_dot = 0
        """

        #step += 1

        ##################################################

        self.state = (x,x_dot,theta,theta_dot,self.goal_x)
        done =  x < self.x_threshold_lower \
                or x > self.x_threshold_upper

        ## Distance to goal
        distance = abs(x - self.goal_x)
        #print("Distance to goal: %.3f " % distance)


        ##################################################

        done = bool(done)
        self.reward = self.reward_calc(distance, theta_dot, theta, x_dot, x)
        #done = False
        if not done:
            reward = self.reward_calc(distance, theta_dot, theta, x_dot, x)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.reward_calc(distance, theta_dot,theta, x_dot, x)
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = self.reward_calc(distance, theta_dot, theta, x_dot, x)

        out_state = (x,theta,x_dot,theta_dot,self.goal_x, self.x_threshold_lower, self.x_threshold_upper)
        #out_state = (x,theta,self.goal_x)

        return np.array(out_state), reward, done, {}


    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        self.state[0] = (self.np_random.random_sample(1) * 2 - 1) * self.x_threshold_upper
        self.state[1] = 0
        self.state[2] = 0
        self.state[3] = 0
        self.state[4] = (self.np_random.random_sample(1) * 2 - 1) * self.x_threshold_upper
        self.steps_beyond_done = None

        self.reward = 0

        self.set_goal()

        # return np.array(self.state)

        out_state = (self.state[0],self.state[2],self.state[3],self.state[1],self.state[4], self.x_threshold_lower, self.x_threshold_upper)
        #out_state = (self.state[0],self.state[2],self.state[4])
        return np.array(out_state)



    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold*2
        #world_width = 3.2
        scale = screen_width/world_width
        carty = 400 # TOP OF CART
        polewidth = 0.04 * scale #10.0
        polelen = 0.7 * scale
        cartwidth = 0.25 * scale #50.0
        cartheight = 0.15 * scale #30.0

        goal_x = self.goal_x;
        goal_size = 20;

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.start = rendering.Line((self.x_threshold_upper*scale+screen_width/2.0,carty),(self.x_threshold_upper*scale+screen_width/2.0,carty+100))
            self.start.set_color(0,0.9,0)
            self.viewer.add_geom(self.start)
            self.limit = rendering.Line((self.x_threshold_lower*scale+screen_width/2.0,carty),(self.x_threshold_lower*scale+screen_width/2.0,carty+100))
            self.limit.set_color(0.9,0,0)
            self.viewer.add_geom(self.limit)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)


            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            #label = rendering.make_text(400, 300, '{:.2f}'.format(self.action))
    		#self.labeltrans = rendering.Transform()
    		#label.add_attr(self.labeltrans)
    		#self.viewer.add_geom(label)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(math.pi+x[2])
        #self.labeltrans.set_text('{:.2f}'.format(self.action))

        label_action = rendering.make_text(200, 550, '{:.2f}'.format(self.action))
        self.viewer.add_onetime(label_action)
        label_reward = rendering.make_text(600, 550, '{:.2f}'.format(self.reward))
        self.viewer.add_onetime(label_reward)
        # Goal line
        self.goal = rendering.Line((goal_x*scale+screen_width/2.0,carty-goal_size), (goal_x*scale+screen_width/2.0,carty+goal_size*3))
        self.goal.set_color(0,0,0)
        self.viewer.add_onetime(self.goal)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
    	if self.viewer:
    		self.viewer.close()
    		self.viewer = None
