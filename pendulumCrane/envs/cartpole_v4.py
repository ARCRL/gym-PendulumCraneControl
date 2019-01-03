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

class CartPoleEnv_Crane3(gym.Env):
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
        self.length = 0.7 
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Motor constant
        R =  1.34             # Terminal resistance - Ohm

        Ke = 2.19*10**(-3)      # Back-EMF constant mV/rpm
        Ke = Ke * 60/(2*math.pi)     # rpm to rad/s
        Kb = Ke               # Back-EMF constant

        Km = 20.9            # Torque constant - mNm/amp
        Km = Km*10**(-3)          # Converted to SI - Nm/amp

        L = 220           # Rotor inductance - uH
        L = L*10**(-6)            # Converted to SI - H

        J = 52                 # Rotor inertia - g*cm^2
        J_m = J*10**(-7)            # Converted to SI - kg*m^2

        Bm = 0.0002    # Viscious friction of motor

        N = 14                 # Gear ratio
        r_wd = 0.03             # Radius of spoole

        cogwheel_gear = 6/4
        N = N*cogwheel_gear

        g = 9.8                # Gravity

        m_p = 0.200            # pendulum mass
        m_c = 2                # Cart mass
        m_all = m_p+m_c

        l_p = 0.78

        J_l = (m_all*r_wd**2)/N**2

        J_tot = J_m + J_l

        b_damp = 0.13

        A_ss = np.array([[-R/L, -Kb/L, 0, 0, 0],\
                        [Km/J_tot, -Bm/J_tot, 0, 0, 0],\
                        [0, r_wd/N, 0, 0, 0],\
                        [-Km/J_tot*r_wd/(N*l_p), r_wd/(N*l_p)*Bm/J_tot, 0, -b_damp/l_p, -g/l_p],\
                        [0, 0, 0, 1, 0]])

        B_ss = np.array([[1/L], [0], [0], [0], [0]])        

        C_ss = np.array([[0,0,1,0,0],[0,0,0,0,180/math.pi]])

        D_ss = np.array([[0], [0]])

        system = signal.StateSpace(A_ss, B_ss, C_ss, D_ss)
        self.system = system.to_discrete(self.tau)

        print(self.system)

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 360 * math.pi / 180
        self.theta_threshold_radians = 120 * math.pi / 180

        # Set threshholds so it won't move outside the physical track
        # UPDATE
        self.x_threshold = 1.0
        self.x_threshold_upper = 1.1
        self.x_threshold_lower = -0.1

        self.goal_x = -0.20

        self.time = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Continuous(20)
        #self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.mul = 1
        self.seed()
        self.viewer = None
        self.state = None
        self.action = self.action_space.sample()
        self.reward = 0

        self.steps_beyond_done = None
        self.reset()

    def set_goal(self, goal=None):
    	if goal == None:
    		goal = self.np_random.random_sample(1)[0]
    	self.goal_x = np.clip(goal, 0, 1)
    	self.state[5] = self.goal_x
    	
    def get_goal(self):
    	return self.goal_x

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reward_calc(self, action, next_state, old_state):

        # Calculate distance 
        x_goal = next_state[2]
        x_pos = next_state[0]
        distance = x_pos - x_goal

        x_goal_old = old_state[2]
        x_pos_old = old_state[0]
        old_distance = x_pos_old - x_goal_old

        # Calculate end effector position of pendulum
        theta = next_state[1]

        pen_length = 0.7 # Length of pendulum - Placeholder
        theta_pos = x_pos + math.sin(theta)*pen_length

        # Use distance of end effector instead of sledge
        distance_pendulum = theta_pos - x_goal

        dist_dif = distance**2 - old_distance**2
        #dist_dif = 
        #print("AR: ", distance, " ", old_distance) 
        reward = 0
        reward_type = 0
        # Base reward:
        if dist_dif > 0:
            reward = -1
        elif dist_dif == 0:
            reward = 1
        elif dist_dif < 0:
            reward = 1


        # Distance bonus
        dist_bonus = 2 - abs(distance)*10
        reward = reward + dist_bonus

        # Theta_pos penalty
        theta_p = next_state[1]
        theta_p = (x_pos+math.sin(theta_p)*0.7)-x_pos
        theta_penalty = abs(theta_p)*2

        reward = reward - theta_penalty
        self.reward = reward
        return reward

    def get_reward(self):
        return self.reward

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        #action = 10

        self.action = action

        i_a, omega_m, x, omega_p, theta_p, goal = state
        x_v = omega_m / 700
        #init_state = (x, theta_p, self.goal_x)
        init_state = (x, x_v, theta_p, omega_p, self.goal_x)

        V_a = action
        #if (abs(action) < 2):
        #    V_a = 0

        old_state = (x, theta_p, self.goal_x)

        xk = np.matmul(self.system.A, np.array([i_a, omega_m, x, omega_p, theta_p])) + self.system.B.T * V_a
        xk = xk[0]

        # Extract state variables
        i_a = xk[0]
        omega_m = xk[1]
        x = xk[2]
        omega_p = xk[3]
        theta_p = xk[4]

        self.state = (i_a, omega_m, x, omega_p, theta_p,self.goal_x)
        done =  x < self.x_threshold_lower \
                or x > self.x_threshold_upper

        ## Distance to goal
        distance = (x - self.goal_x)
        self.x_tmp = x
        #distance = x - math.sin(theta_p)*self.length - self.goal_x
        ##################################################

        done = bool(done)

        if done:
            if self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                self.steps_beyond_done += 1

        x_v = omega_m / 700
        
        #out_state = (x, theta_p, self.goal_x)
        #print(self.goal_x)

        self.time += self.tau
        
        out_state = (x, theta_p, self.goal_x)
        new_state = out_state
        #out_state = (x, theta_p, self.goal_x)

        self.reward = self.reward_calc(action, new_state, old_state)
        reward = self.reward

        return np.array(out_state), reward, done, {}


    def reset(self, f='log.txt'):
        self.state = np.zeros(6)
        self.state[0] = 0#self.np_random.random_sample(1) * self.x_threshold
        self.state[1] = 0
        self.state[2] = self.np_random.random_sample(1)
        self.state[3] = 0
        self.state[4] = 0 #math.pi
        self.state[5] = 0
        self.steps_beyond_done = None

        self.reward = 0

        self.time = 0

        self.set_goal()

        out_state = (self.state[2], self.state[4], self.state[5])
        #out_state = (self.state[2], self.state[4], self.state[5])
        return np.array(out_state)


    def render(self, mode='human'):
        self.state = np.zeros(6)
        self.state[0] = 0
        self.state[1] = 0
        #self.state[2] = self.np_random.random_sample(1)
        self.state[2] = np.random.randint(0,high=2,size=1)
        #self.state[2] = 0.5
        self.state[3] = 0
        #self.state[4] = np.random.uniform(low=-30*math.pi/180,high=30*math.pi/180,size=(1,1))
        self.state[5] = 0
        self.steps_beyond_done = None

        self.reward = 0

        self.time = 0

        self.action = 0

        self.set_goal()

        #out_state = (self.state[2],self.state[4],self.state[2],self.state[4],self.state[5])
        
        #out_state = (self.state[2],self.state[4], self.action, self.state[5])

        out_state = (self.state[2],self.state[4],self.state[5])

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

            self.start = rendering.Line(((self.x_threshold_upper-0.6)*scale+screen_width/2.0,carty),((self.x_threshold_upper-0.6)*scale+screen_width/2.0,carty+100))
            self.start.set_color(0,0.9,0)
            self.viewer.add_geom(self.start)
            self.limit = rendering.Line(((self.x_threshold_lower-0.4)*scale+screen_width/2.0,carty),((self.x_threshold_lower-0.4)*scale+screen_width/2.0,carty+100))
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
        cartx = (0.5-x[2])*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(math.pi-x[4])
        #self.labeltrans.set_text('{:.2f}'.format(self.action))

        label_action = rendering.make_text(190, 150, 'Action={:.2f}'.format(self.action))
        self.viewer.add_onetime(label_action)
        label_reward = rendering.make_text(190, 80, 'Reward={:.2f}'.format(self.reward))
        self.viewer.add_onetime(label_reward)
        label_time = rendering.make_text(400, 550, 'Time={:.2f}'.format(self.time))
        self.viewer.add_onetime(label_time)

        label_x = rendering.make_text(190, 220, 'Dist={:.2f}'.format(x[2]-self.goal_x))
        #label_x = rendering.make_text(190, 220, 'Dist={:.2f}'.format(self.pend_dist))
        self.viewer.add_onetime(label_x)

        # Goal line
        self.goal = rendering.Line(((0.5-goal_x)*scale+screen_width/2.0,carty-goal_size), ((0.5-goal_x)*scale+screen_width/2.0,carty+goal_size*3))
        self.goal.set_color(0,0,0)
        self.viewer.add_onetime(self.goal)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
