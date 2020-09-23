from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import random
import time


class TerranBasicAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranBasicAgent, self).__init__()

        self.base_top_left = None
        self.supply_depot_built = False
        self.barracks_built = False
        self.barracks_rallied = False
        self.army_rallied = False

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
        super(TerranBasicAgent, self).step(obs)

        time.sleep(0.5)

        if obs.first():
            self.base_top_left = None
            self.supply_depot_built = False
            self.barracks_built = False
            self.barracks_rallied = False
            self.army_rallied = False

            player_y, player_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        if not self.supply_depot_built:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                    ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                    if len(ccs) > 0:
                        mean_x, mean_y = self.getMeanLocation(ccs)
                        target = self.transformLocation(int(mean_x), 0, int(mean_y), 20)
                        self.supply_depot_built = True

                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)
            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                scv = random.choice(scvs)
                return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                 scv.y))
        elif not self.barracks_built:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                    ccs = self.get_units_by_type(obs, units.Terran.CommandCenter)
                    if len(ccs) > 0:
                        mean_x, mean_y = self.getMeanLocation(ccs)
                        target = self.transformLocation(int(mean_x), 20, int(mean_y), 0)
                        self.barracks_built = True

                        return actions.FUNCTIONS.Build_Barracks_screen("now", target)
            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                scv = random.choice(scvs)
                return actions.FUNCTIONS.select_point("select", (scv.x,
                                                                 scv.y))

        elif not self.barracks_rallied:
            if self.unit_type_is_selected(obs, units.Terran.Barracks):
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 21])
                else:
                    return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 46])
            barracks = self.get_units_by_type(obs, units.Terran.Barracks)
            if len(barracks) > 0:
                barrack = random.choice(barracks)
                return actions.FUNCTIONS.select_point("select", (barrack.x,
                                                                 barrack.y))
        elif obs.observation.player.food_cap - obs.observation.player.food_used:
            if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                return actions.FUNCTIONS.Train_Marine_quick("queued")

        elif not self.army_rallied:
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                self.army_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Attack_minimap("now", [39, 45])
                else:
                    return actions.FUNCTIONS.Attack_minimap("now", [21, 24])

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()
