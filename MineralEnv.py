import numpy as np

from pysc2.env import sc2_env, environment
from pysc2.lib import actions

# How to set up custom tf-agents envrionment
# https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059

class MineralEnv():
    # Default settings for initializing CollectMineralShards
    metadata = {'render.modes': ['human']}
    default_settings = {
    'map_name': "CollectMineralShards",
    'players': [sc2_env.Agent(sc2_env.Race.terran)],
    'agent_interface_format': sc2_env.parse_agent_interface_format(
        feature_screen=84,
        feature_minimap=64,
        action_space=None,
        use_feature_units=False,
        use_raw_units=False),
    'realtime': False,
    'visualize': True,
    'disable_fog': True,
    }

    def __init__(self):
        self.obs_shape = (1, 84, 84)
        self._episode_ended = False
        self.env = None

    def reset(self):
        self._episode_ended = False

        if self.env == None:
            args = {**self.default_settings}
            self.env = sc2_env.SC2Env(**args)
        
        raw_obs = self.env.reset()[0]
        # Grab all marines
        self.env.step([actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])])
        feature_screen = self._get_feature_screen(raw_obs)
        return np.array(feature_screen)

    def _get_feature_screen(self, raw_obs):
        obs = raw_obs.observation["feature_screen"][5]
        return np.reshape(obs, self.obs_shape)

    def step(self, action):
        if self._episode_ended:
            return self._reset()
        
        x = action // 84
        y = action % 84
        raw_obs = self.env.step(
            [actions.FunctionCall(
                actions.FUNCTIONS.Attack_screen.id, [[0], [x,y]])
            ]
        )[0]

        feature_screen = self._get_feature_screen(raw_obs)
        self._episode_ended = raw_obs.step_type == environment.StepType.LAST
        return np.array(feature_screen), raw_obs.reward, self._episode_ended, {}

    def close(self):
        self.env.close()
