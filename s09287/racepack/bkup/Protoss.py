from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import random
import time


class ProtossBasicAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossBasicAgent, self).__init__()

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

    def get_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def step(self, obs):
        super(ProtossBasicAgent, self).step(obs)

        time.sleep(0.5)

        if obs.first():
            self.base_top_left = None
            self.pylon_built = False
            self.gateway_built = False
            self.gateway_rallied = False
            self.army_rallied = False

            player_y, player_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        supply = obs.observation.player.food_cap - obs.observation.player.food_used
        completed_pylons = self.get_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_units_by_type(obs, units.Protoss.Gateway)

        if not self.pylon_built:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
                    nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
                    if len(nexus) > 0:
                        mean_x, mean_y = self.getMeanLocation(nexus)
                        target = self.transformLocation(int(mean_x), 0, int(mean_y), 20)
                        self.pylon_built = True

                        return actions.FUNCTIONS.Build_Pylon_screen("now", target)
            if self.can_do(obs, actions.FUNCTIONS.select_idle_worker.id):
                return actions.FUNCTIONS.select_idle_worker("select")
            elif len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                 probe.y))

        elif (len(completed_pylons) > 0 and not self.gateway_built and
                obs.observation.player.minerals >= 150):
            if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
                mean_x, mean_y = self.getMeanLocation(completed_pylons)
                target = self.transformLocation(int(mean_x), 10, int(mean_y), 0)
                self.gateway_built = True
                return actions.FUNCTIONS.Build_Gateway_screen("now", target)
            probes = self.get_units_by_type(obs, units.Protoss.Probe)

            if self.can_do(obs, actions.FUNCTIONS.select_idle_worker.id):
                return actions.FUNCTIONS.select_idle_worker("select")
            elif len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                 probe.y))

        # elif len(probes) < 13 or len(probes) < obs.observation.player.food_cap* 0.5:
        #     nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
        #     if self.can_do(obs, actions.FUNCTIONS.Train_Probe_quick.id):
        #         return actions.FUNCTIONS.Train_Probe_quick("queued")
        #     if len(nexus)>0:
        #         nex = random.choice(nexus)
        #         return actions.FUNCTIONS.select_point("select", (nex.x, nex.y))

        elif supply > 1:
            if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
                return actions.FUNCTIONS.Train_Zealot_quick("queued")
            if len(gateways) > 0:
                gateway = random.choice(gateways)
                return actions.FUNCTIONS.select_point("select", (gateway.x, gateway.y))

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