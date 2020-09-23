import random
import time
import math

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_COCOON = 'selectcocoon'
ACTION_BUILD_SUPPLY_EXTRACTOR = 'buildextractor'
ACTION_BUILD_CREEPTUMOR = 'buildcreeptumor'
ACTION_SELECT_CREEPTUMOR = 'selectcreeptumor'
ACTION_BUILD_ZERGLING = 'buildzergling'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_COCOON,
    ACTION_BUILD_SUPPLY_EXTRACTOR,
    ACTION_BUILD_CREEPTUMOR,
    ACTION_SELECT_CREEPTUMOR,
    ACTION_BUILD_ZERGLING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


# reference from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            # state_action = self.q_table.ix[observation, :]
            state_action = self.q_table.loc[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # q_predict = self.q_table.ix[s, a]
        q_predict = self.q_table.loc[s, a]
        # q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        # update
        # self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class ZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergAgent, self).__init__()

        self.base_top_left = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        super(ZergAgent, self).step(obs)

        # time.sleep(0.5)

        if obs.first():
            player_y, player_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        extractor_count = len(self.get_units_by_type(obs, units.Zerg.Extractor))

        creeptumor_count = len(self.get_units_by_type(obs, units.Zerg.CreepTumor))

        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_used

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures

        current_state = [
            extractor_count,
            creeptumor_count,
            supply_limit,
            army_supply,
        ]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_COCOON:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                cocoons = self.get_units_by_type(obs, units.Zerg.Cocoon)
                if len(cocoons) > 0:
                    cocoon = random.choice(cocoons)
                    if cocoon.x >= 0 and cocoon.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (cocoon.x,
                                                                         cocoon.y))

        elif smart_action == ACTION_BUILD_SUPPLY_EXTRACTOR:
            if self.can_do(obs, actions.FUNCTIONS.Build_Extractor_screen.id):
                ccs = self.get_units_by_type(obs, units.Zerg.Extractor)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transformLocation(int(mean_x), 0, int(mean_y), 20)

                    return actions.FUNCTIONS.Build_Extractor_screen("now", target)

        elif smart_action == ACTION_BUILD_CREEPTUMOR:
            if self.can_do(obs, actions.FUNCTIONS.Build_CreepTumor_screen.id):
                ccs = self.get_units_by_type(obs, units.Zerg.CreepTumor)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transformLocation(int(mean_x), 20, int(mean_y), 0)

                    return actions.FUNCTIONS.Build_CreepTumor_screen("now", target)

        elif smart_action == ACTION_SELECT_CREEPTUMOR:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                CreepTumors = self.get_units_by_type(obs, units.Zerg.CreepTumor)
                if len(CreepTumors) > 0:
                    CreepTumor = random.choice(CreepTumors)
                    if CreepTumor.x >= 0 and CreepTumor.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (CreepTumor.x,
                                                                         CreepTumor.y))

        elif smart_action == ACTION_BUILD_ZERGLING:
            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("queued")

        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                if self.base_top_left:
                    return actions.FUNCTIONS.Attack_minimap("now", [39, 45])
                else:
                    return actions.FUNCTIONS.Attack_minimap("now", [21, 24])

        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    agent = ZergAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    #map_name="AbyssalReef",
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                      feature_dimensions=features.Dimensions(screen=84, minimap=64),
                      use_feature_units=True),
                    step_mul=8,
                    game_steps_per_episode=0,
                    visualize=True) as env:

              agent.setup(env.observation_spec(), env.action_spec())

              timesteps = env.reset()
              agent.reset()

              while True:
                  step_actions = [agent.step(timesteps[0])]
                  if timesteps[0].last():
                      break
                  timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
