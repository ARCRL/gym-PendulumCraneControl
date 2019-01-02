import argparse
import sys

import pendulumCrane
import gym
from gym import wrappers, logger
from sklearn.gaussian_process.kernels import WhiteKernel

import numpy as np

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import random
import math

class ddpgAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env, action_space):
        self.env = env
        self.action_space = action_space
        learning_rate = 0.001

        self.load = 0
        self.save = 0
        self.training = 0
        
        self.state_size = 5
        self.current_critic = critic(self.state_size + 1, 30, 1, learning_rate)
        self.target_critic = critic(self.state_size + 1, 30, 1, learning_rate)
        self.target_critic.load_state_dict(self.current_critic.state_dict())

        self.current_actor = actor(self.state_size, 30, 1, learning_rate)
        self.target_actor = actor(self.state_size, 30, 1, learning_rate)
        self.target_actor.load_state_dict(self.current_actor.state_dict())

        if self.load:
            print("Loading saved state dicts")
            self.current_actor.load_state_dict(torch.load('./current_actor.pt'))
            self.target_actor.load_state_dict(torch.load('./target_actor.pt'))
            self.current_critic.load_state_dict(torch.load('./current_critic.pt'))
            self.target_critic.load_state_dict(torch.load('./target_critic.pt'))


        self.replay_memory_capacity = 100000

        self.replay_memory = ReplayMemory(self.replay_memory_capacity)
        print('state size: ', self.state_size)

    def train(self):
        epochs = 20
        gamma = 0.99 # discount rate
        tau = 0.01 # target network update rate
        batch_size = 64
        prefill_memory = True

        count = 1

        epoch_steps = 2000

        noiseProcess_std = 0.1

        loss_hist = np.zeros(1)
        p_loss_hist = np.zeros(1)
        reward = np.zeros(1)

        if prefill_memory:
            print('prefill replay memory')
            s = env.reset()
            while self.replay_memory.count() < self.replay_memory_capacity:
                a = env.action_space.sample()
                s1, r, d, _ = env.step(a)
                s1 = s1[0]
                self.replay_memory.add(s, a, r, s1, d)
                s = s1
                if d:
                    s = env.reset()

        batch = np.array(self.replay_memory.sample(2000))#,dtype=float)

        # Extract from batch
        ss, aa, rr, ss1, dd = np.stack(batch[:,0]), np.stack(batch[:,1]), np.stack(batch[:,2]), np.stack(batch[:,3]), np.stack(batch[:,4]).astype(int)

        # Convert to Tensors
        ss = (ss).reshape(-1, self.state_size)
        aa = (aa).reshape(-1,1)
        rr = (rr).reshape(-1,1)
        ss1 = (ss1).reshape(-1,self.state_size)
        dd = (dd).reshape(-1,1)

        fig, ax = plt.subplots(5,1)
        ax[0].plot(range(2000), ss)
        ax[0].grid()
        ax[1].plot(range(2000), aa)
        ax[1].grid()
        ax[2].plot(range(2000), rr)
        ax[2].grid()
        ax[3].plot(range(2000), ss1)
        ax[3].grid()
        ax[4].plot(range(2000), dd)
        ax[4].grid()
        #plt.show()

        for e in range(epochs):
            print("Starting epoch {}".format(e))
            s = env.reset()
            env.env.mul = (e+1)*2#(e + 1)/2
            for j in range(epoch_steps):
                with torch.no_grad():
                    a = self.current_actor(torch.from_numpy(s).float()).numpy() + np.random.normal(0, noiseProcess_std, 1)
                a = np.clip(a,-10,10)
                # Step with new action and save to memory
                s1, r, d, _ = self.env.step(a[0])
                s1 = s1[0]
                self.replay_memory.add(s, a[0], r, s1, d)
                
                reward = np.append(reward,r)

                # Update step
                if self.replay_memory.count() >= batch_size:
                    # sample batch from replay memory
                    batch = np.array(self.replay_memory.sample(batch_size))#,dtype=float)

                    # Extract from batch
                    ss, aa, rr, ss1, dd = np.stack(batch[:,0]), np.stack(batch[:,1]), np.stack(batch[:,2]), np.stack(batch[:,3]), np.stack(batch[:,4]).astype(int)

                    # Convert to Tensors
                    ss = torch.from_numpy(ss).float().view(-1,self.state_size)
                    aa = torch.from_numpy(aa).float().view(-1,1)
                    rr = torch.from_numpy(rr).float().view(-1,1)
                    ss1 = torch.from_numpy(ss1).float().view(-1,self.state_size)
                    dd = torch.from_numpy(dd).float().view(-1,1)


                    #with torch.no_grad():
                    aa1 = self.target_actor(ss1)

                    Qt_in = torch.cat((ss1, aa1),1)
                    Qt = self.target_critic(Qt_in)


                    self.current_critic.optimizer.zero_grad()

                    y = rr + gamma * Qt * dd

                    Qc_in = torch.cat([ss,aa],1)
                    Qc = self.current_critic(Qc_in)

                    ## Update critic
                    loss = self.current_critic.loss(y,Qc)
                    #loss1 = self.current_critic.MSEloss(Qc,y)/64

                    loss.backward()

                    self.current_critic.optimizer.step()
                    self.target_critic.update_params(self.current_critic.state_dict(), tau)

                    ## Update actor
                    self.current_actor.optimizer.zero_grad()

                    aa = self.current_actor(ss)

                    Qc_in = torch.cat((ss, aa),1)
                    Qc = -self.current_critic(Qc_in)

                    Qc = torch.mean(Qc)
                    Qc.backward()

                    self.current_actor.optimizer.step()
                    self.target_actor.update_params(self.current_actor.state_dict(), tau)

                    loss_hist = np.append(loss_hist,loss.detach().numpy())
                    p_loss_hist = np.append(p_loss_hist,Qc.detach().numpy())
                
                    count += 1
                s = s1
                if j%100 == 0: print("Epoch {}/{}: step {}/{}\nReward = {}\nAction = {}\nDistance = {}\nTarget = {}\nX = {}\nX = {}\n".format(e,epochs,j,epoch_steps,r,a,np.abs(s1[0]-s1[2]),s1[2],s[0], s[1]))
                if d:
                    print("Done reached\nX = {}\n".format(s[0]))
                    break
            if self.save:
                print("Saving models")
                torch.save(self.current_actor.state_dict(), './current_actor_new.pt')
                torch.save(self.current_critic.state_dict(), './current_critic_new.pt')
                torch.save(self.target_critic.state_dict(), './target_critic_new.pt')
                torch.save(self.target_actor.state_dict(), './target_actor_new.pt')


        fig, ax = plt.subplots(3,1)
        ax[0].plot(range(count), loss_hist)
        ax[0].set_xlabel("Steps")
        ax[0].set_ylabel("Value loss (critic)")
        ax[0].grid()

        ax[1].plot(range(count), p_loss_hist)
        ax[1].set_xlabel("Steps")
        ax[1].set_ylabel("Policy loss (actor)")
        ax[1].grid()

        ax[2].plot(range(count), reward)
        ax[2].set_xlabel("Steps")
        ax[2].set_ylabel("Rewards")
        ax[2].grid()

        plt.show()

        batch = np.array(self.replay_memory.sample(2000))#,dtype=float)

        # Extract from batch
        ss, aa, rr, ss1, dd = np.stack(batch[:,0]), np.stack(batch[:,1]), np.stack(batch[:,2]), np.stack(batch[:,3]), np.stack(batch[:,4]).astype(int)

        # Convert to Tensors
        ss = (ss).reshape(-1,self.state_size)
        aa = (aa).reshape(-1,1)
        rr = (rr).reshape(-1,1)
        ss1 = (ss1).reshape(-1,self.state_size)
        dd = (dd).reshape(-1,1)

        fig, ax = plt.subplots(5,1)
        ax[0].plot(range(2000), ss)
        ax[0].grid()
        ax[1].plot(range(2000), aa)
        ax[1].grid()
        ax[2].plot(range(2000), rr)
        ax[2].grid()
        ax[3].plot(range(2000), ss1)
        ax[3].grid()
        ax[4].plot(range(2000), dd)
        ax[4].grid()
        #plt.show()

        print(self.current_actor.hidden.weight)
        #print(self.current_actor.hidden2.weight)

    def act(self, observation, reward, done):
        with torch.no_grad():
            action = self.current_actor(torch.from_numpy(observation).float())
        return np.clip(action.numpy()[0],-10,10)

class ReplayMemory(object):
    """Experience Replay Memory"""
    
    def __init__(self, capacity):
        #self.size = size
        self.memory = deque(maxlen=capacity)
    
    def add(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])
    
    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)
    
    def count(self):
        return len(self.memory)

class actor(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(actor, self).__init__()
        # network
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        #self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_outputs)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-8)

    def forward(self, x):
        #x = F.instance_norm(x)
        x = self.hidden(x)
        x = F.relu(x)
        #x = F.instance_norm(x)
        x = self.hidden2(x)
        x = F.relu(x)
        #x = self.hidden3(x)
        #x = F.relu(x)
        #x = F.instance_norm(x)
        x = self.out(x)
        # x = F.tanh(x)
        return torch.tanh(x)*10
        #return x


    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

class critic(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate):
        super(critic, self).__init__()
        # network
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_outputs)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-8)

        self.MSEloss = nn.MSELoss(reduction='elementwise_mean')

    def forward(self, x):
        #x = F.instance_norm(x)
        #print(x)
        #print(self.hidden.weight)
        x = self.hidden(x)
        #print(x)
        x = F.relu(x)
        #print(x)
        #x = F.batch_norm(x)
        x = self.hidden2(x)
        x = F.relu(x)
        #x = F.batch_norm(x)
        x = self.out(x)
        return x

    def loss(self, q_outputs, q_targets):
        return torch.mean(torch.pow(q_targets - q_outputs, 2))

    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPoleCraneTrain-v2', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)
    env.env.set_goal(0)
    
    agent = ddpgAgent(env, env.action_space)
    agent.train()
    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPoleCrane-v2', help='Select the environment to run')
    args = parser.parse_args()


    env = gym.make(args.env_id)

    outdir = '/tmp/ddpg-agent-results'
    #env = wrappers.Monitor(env, directory=outdir, force=True)

    env.env.mul = 4
    env.seed(1234)

    episode_count = 10#2097865
    reward = 0
    done = False	

    for i in range(episode_count):
        ob = env.reset()
        ob = env.env.reset('ddpg_' + str(i) +'.txt')
        #env.env.env.state[2] = 0.05
        #env.env.env.set_goal(0.8)
        #while True:
        env.env.log()
        print(ob[0], ob[-1])
        for j in range(1500):
            action = agent.act(ob, reward, done)
            action = np.clip(action, -10, 10)
            #action = -10.0
            ob, reward, done, _ = env.step(action)
            ob = ob[0]
            env.env.log()
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.env.close()
