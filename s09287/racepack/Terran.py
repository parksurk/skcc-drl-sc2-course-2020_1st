from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from DwseoRLAgent import QLearningTable
import pandas as pd
import numpy as np
import random
import time
import os
import math


DATA_FILE = 'rlagent_learning_data_terran'

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
]

#for mm_x in range(0, 64):
#    for mm_y in range(0, 64):
#        smart_actions.append(ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y))

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


class TerranRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranRLAgent, self).__init__()

        self.base_top_left = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

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
        super(TerranRLAgent, self).step(obs)

        #time.sleep(0.5)

        if obs.last():
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        supply_depot_count = len(self.get_units_by_type(obs, units.Terran.SupplyDepot))

        barracks_count = len(self.get_units_by_type(obs, units.Terran.Barracks))

        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_used

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures

#        current_state = np.zeros(5000)
#        current_state[0] = supply_depot_count
#        current_state[1] = barracks_count
#        current_state[2] = supply_limit
#        current_state[3] = army_supply
#
#        hot_squares = np.zeros(4096)
#        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
#        for i in range(0, len(enemy_y)):
#            y = int(enemy_y[i])
#            x = int(enemy_x[i])
#
#            hot_squares[((y - 1) * 64) + (x - 1)] = 1
#
#        if not self.base_top_left:
#            hot_squares = hot_squares[::-1]
#
#        for i in range(0, 4096):
#            current_state[i + 4] = hot_squares[i]

        current_state = np.zeros(20)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

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

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_SCV:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                if len(scvs) > 0:
                    scv = random.choice(scvs)
                    if scv.x >= 0 and scv.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                              scv.y))

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transformDistance(int(mean_x), 0, int(mean_y), 20)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

        elif smart_action == ACTION_BUILD_BARRACKS:
            if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    target = self.transformDistance(int(mean_x), 20, int(mean_y), 0)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", target)

        elif smart_action == ACTION_SELECT_BARRACKS:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                barracks = self.get_units_by_type(obs, units.Terran.Barracks)
                if len(barracks) > 0:
                    barrack = random.choice(barracks)
                    if barrack.x >= 0 and barrack.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (barrack.x,
                                                                              barrack.y))

        elif smart_action == ACTION_BUILD_MARINE:
            if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                return actions.FUNCTIONS.Train_Marine_quick("queued")

        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            #if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
            if not self.unit_type_is_selected(obs, units.Terran.SCV) and self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(x), int(y)))

        return actions.FUNCTIONS.no_op()
