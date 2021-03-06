from collections import defaultdict
import numpy as np

from pysc2.env import sc2_env, environment
from pysc2.lib import actions

from absl import flags
# How to set up custom tf-agents envrionment
# https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059

# Flags needed for creating pysc2 environment
FLAGS = flags.FLAGS
FLAGS([''])

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

    def __init__(self, realtime, visualize):
        self.obs_shape = (1, 84, 84)
        self._episode_ended = False
        self.env = None
        self.available_actions = None

        self.default_settings["realtime"] = realtime
        self.default_settings["visualize"] = visualize

    def reset(self):
        self._episode_ended = False

        if self.env == None:
            args = {**self.default_settings}
            self.env = sc2_env.SC2Env(**args)
        
        raw_obs = self.env.reset()[0]
        # Grab all marines
        self.env.step([actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])])
        feature_screen = self._get_feature_screen(raw_obs)
        self.available_actions = None if "available_actions" not in raw_obs.observation.keys() else raw_obs.observation["available_actions"]
        return np.array(feature_screen)

    def _get_feature_screen(self, raw_obs):
        obs = raw_obs.observation["feature_screen"][5]
        return np.reshape(obs, self.obs_shape)

    def step(self, action):
        x = action // 84
        y = action % 84

        # Take main action
        total_reward = 0
        done = False
        raw_obs = None
        if len(self.available_actions) and actions.FUNCTIONS.Attack_screen.id in self.available_actions:
            raw_obs, reward, done = self.take_move(x, y)
            total_reward += reward
        else:
            raw_obs, reward, done = self.take_noop()
            total_reward += reward
        if done:
            self.available_actions = None if "available_actions" not in raw_obs.observation.keys() else raw_obs.observation["available_actions"]
            return np.array(self._get_feature_screen(raw_obs)), total_reward, done, {}

        # Take 7 more steps
        for _ in range(6):
            raw_obs, reward, done = self.take_noop()
            total_reward += reward

            if done:
                self.available_actions = None if "available_actions" not in raw_obs.observation.keys() else raw_obs.observation["available_actions"]
                return np.array(self._get_feature_screen(raw_obs)), total_reward, done, {}

        raw_obs, reward, done = self.take_noop()
        total_reward += reward

        self.available_actions = None if "available_actions" not in raw_obs.observation.keys() else raw_obs.observation["available_actions"]
        feature_screen = self._get_feature_screen(raw_obs)

        return np.array(feature_screen), total_reward, done, {}

    def close(self):
        self.env.close()

    def take_noop(self):
        raw_obs = self.env.step([actions.FunctionCall(
                actions.FUNCTIONS.no_op.id, []
                )])[0]
        done = raw_obs.step_type == environment.StepType.LAST
        return raw_obs, raw_obs.reward, done

    def take_move(self, x, y):
        raw_obs = self.env.step(
            [actions.FunctionCall(
                actions.FUNCTIONS.Move_screen.id, [[0], [x,y]])
            ]
        )[0]
        done = raw_obs.step_type == environment.StepType.LAST
        return raw_obs, raw_obs.reward, done