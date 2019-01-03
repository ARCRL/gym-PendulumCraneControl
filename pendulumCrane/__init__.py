from gym.envs.registration import register
from pendulumCrane.agents.ddpg_agent import ddpgAgent

register(
    id='CartPoleCrane-v0',
    entry_point='pendulumCrane.envs:CartPoleEnv_Crane',
    max_episode_steps=10000,
    reward_threshold=2000,
)

register(
    id='CartPoleCrane-v1',
    entry_point='pendulumCrane.envs:CartPoleEnv_Crane',
    max_episode_steps=800,
    reward_threshold=2000,
)

register(
    id='CartPoleCrane-v2',
    entry_point='pendulumCrane.envs:CartPoleEnv_Crane2',
    max_episode_steps=600,
    reward_threshold=2000,
)

register(
    id='CartPoleCraneTrain-v2',
    entry_point='pendulumCrane.envs:CartPoleEnv_Crane2',
    max_episode_steps=2000,
    reward_threshold=2000,
)

register(
    id='CartPoleCrane-v3',
    entry_point='pendulumCrane.envs:CartPoleEnv_Crane3',
    max_episode_steps=600,
    reward_threshold=2000,
)

register(
    id='CartPoleCraneTrain-v3',
    entry_point='pendulumCrane.envs:CartPoleEnv_Crane3',
    max_episode_steps=2000,
    reward_threshold=2000,
)