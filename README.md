# gym-PendulumCraneControl
This was part of the assessment in course 02456 - Deep Learning at DTU.

The environments are based on a rig used in course 31340 - Digital Control at DTU. Which is a pendulum crane consisting of a actuated cart on a linear rail with an unactuated rigid pendulum with a pickup mechanisim attached to the end. For simplicity only the cart and pendulum are modeled. Only the speed of the cart is controlled through the voltage to a DC motor which accepts +/-10 V. 

In the Digital Control course an LQR with integral action (state space controller) was designed and the aim of this project was to evaluate different Reinforcement Learning algorithms to this optimal controller to see if is was possible to create an agent with comparable or better performance with no or very limited prior knowledge of the system.

DQN and DDPG was tested againts the LQR and two very simple baseline agents which moves with constant speed towards the goal and stops once the goal is passed; one with +/-10 V (maximumspeed) and another +/-2 V.

As expected the DDPG, which is a continuous agent, outperforms the DQN, which is a discrete agent limited +/- [0, 0.1, 0.2, 1.0, 2.0, 5.0, 7.0, 9.0] V outputs, and both simple agents. The LQR however was still best.

# Showcase
Two notebooks has been created to showcase the results obtained during the project; one for DQN and one for DDPG.

# Installation
The installation requires a functioning version of OpenAI gym and PyTorch.
```bash
cd gym-PendulumCraneControl
pip install -e .
```