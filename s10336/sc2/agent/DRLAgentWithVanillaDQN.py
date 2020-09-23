import random
import time
import math
import os.path

import numpy as np
import pandas as pd
from collections import deque
import pickle

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units, upgrades
from absl import app

import torch
from torch.utils.tensorboard import SummaryWriter

from s10336.skdrl.pytorch.model.mlp import NaiveMultiLayerPerceptron
from s10336.skdrl.common.memory.memory import ExperienceReplayMemory
from s10336.skdrl.pytorch.model.dqn import DQN, prepare_training_inputs


DATA_FILE_QNET = 's10336_rlagent_with_vanilla_dqn_qnet'
DATA_FILE_QNET_TARGET = 's10336_rlagent_with_vanilla_dqn_qnet_target'
SCORE_FILE = 's10336_rlagent_with_vanilla_dqn_score'

scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#writer = SummaryWriter()

class TerranAgentWithRawActsAndRawObs(base_agent.BaseAgent):
    # actions 추가 및 함수 정의(hirerachy하게)

    actions = ("do_nothing",
               "train_scv",
               "harvest_minerals",
               "harvest_gas",
               "build_commandcenter",

               "build_refinery",
               "build_supply_depot",
               "build_barracks",
               "train_marine",

               "build_factorys",
               "build_techlab_factorys",
               "train_tank",

               "build_armorys",

               "build_starports",
               "build_techlab_starports",
               "train_banshee",

               "attack",
               "attack_all",

               "tank_control"
               )

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_my_units_by_type(self, obs, unit_type):
        if unit_type == units.Neutral.VespeneGeyser:  # 가스 일 때만
            return [unit for unit in obs.observation.raw_units
                    if unit.unit_type == unit_type]

        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(TerranAgentWithRawActsAndRawObs, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
            self.top_left_gas_xy = [(14, 25), (21, 19), (46, 23), (39, 16)]
            self.bottom_right_gas_xy = [(44, 43), (37, 50), (12, 46), (19, 53)]

            self.cloaking_flag = 1

            self.TerranVehicleWeaponsLevel1 = False
            self.TerranVehicleWeaponsLevel2 = False
            self.TerranVehicleWeaponsLevel3 = False

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def train_scv(self, obs):
        completed_commandcenterses = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if (len(completed_commandcenterses) > 0 and obs.observation.player.minerals >= 100
                and len(scvs) < 35):
            commandcenters = self.get_my_units_by_type(obs, units.Terran.CommandCenter)

            ccs = [commandcenter for commandcenter in commandcenters if commandcenter.assigned_harvesters < 18]

            if ccs:
                ccs = ccs[0]
                if ccs.order_length < 5:
                    return actions.RAW_FUNCTIONS.Train_SCV_quick("now", ccs.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        commandcenters = self.get_my_units_by_type(obs, units.Terran.CommandCenter)  # 최적 자원 할당 유닛 구현

        cc = [commandcenter for commandcenter in commandcenters if commandcenter.assigned_harvesters < 18]

        if cc:
            cc = cc[0]

            idle_scvs = [scv for scv in scvs if scv.order_length == 0]

            if len(idle_scvs) > 0 and cc.assigned_harvesters < 18:
                mineral_patches = [unit for unit in obs.observation.raw_units
                                   if unit.unit_type in [
                                       units.Neutral.BattleStationMineralField,
                                       units.Neutral.BattleStationMineralField750,
                                       units.Neutral.LabMineralField,
                                       units.Neutral.LabMineralField750,
                                       units.Neutral.MineralField,
                                       units.Neutral.MineralField750,
                                       units.Neutral.PurifierMineralField,
                                       units.Neutral.PurifierMineralField750,
                                       units.Neutral.PurifierRichMineralField,
                                       units.Neutral.PurifierRichMineralField750,
                                       units.Neutral.RichMineralField,
                                       units.Neutral.RichMineralField750
                                   ]]
                scv = random.choice(idle_scvs)
                distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
                mineral_patch = mineral_patches[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_gas(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        refs = self.get_my_units_by_type(obs, units.Terran.Refinery)

        refs = [refinery for refinery in refs if refinery.assigned_harvesters < 3]

        if refs:
            ref = refs[0]
            if len(scvs) > 0 and ref.ideal_harvesters:
                scv = random.choice(scvs)
                distances = self.get_distances(obs, refs, (scv.x, scv.y))
                ref = refs[np.argmin(distances)]

                return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                    "now", scv.tag, ref.tag)

        return actions.RAW_FUNCTIONS.no_op()

    def build_commandcenter(self, obs):
        commandcenters = self.get_my_units_by_type(obs, units.Terran.CommandCenter)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if len(commandcenters) == 0 and obs.observation.player.minerals >= 400 and len(scvs) > 0:
            # 본진 commandcenter가 파괴된 경우
            ccs_xy = (19, 23) if self.base_top_left else (39, 45)
            distances = self.get_distances(obs, scvs, ccs_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                "now", scv.tag, ccs_xy)

        if (len(commandcenters) < 2 and obs.observation.player.minerals >= 400 and
                len(scvs) > 0):
            ccs_xy = (41, 21) if self.base_top_left else (17, 48)

            if len(commandcenters) == 1 and ((commandcenters[0].x, commandcenters[0].y) == (41, 21) or
                                             (commandcenters[0].x, commandcenters[0].y) == (17, 48)):
                # 본진 commandcenter가 파괴된 경우
                ccs_xy = (19, 23) if self.base_top_left else (39, 45)

            distances = self.get_distances(obs, scvs, ccs_xy)
            scv = scvs[np.argmin(distances)]

            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                "now", scv.tag, ccs_xy)
        return actions.RAW_FUNCTIONS.no_op()

    ################################################################################################
    ####################################### refinery ###############################################

    def build_refinery(self, obs):
        refinerys = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if (obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            gas = self.get_my_units_by_type(obs, units.Neutral.VespeneGeyser)[0]

            if self.base_top_left:
                gases = self.top_left_gas_xy
            else:
                gases = self.bottom_right_gas_xy

            rc = np.random.choice([0, 1, 2, 3])
            gas_xy = gases[rc]
            if (gas.x, gas.y) == gas_xy:
                distances = self.get_distances(obs, scvs, gas_xy)
                scv = scvs[np.argmin(distances)]

                return actions.RAW_FUNCTIONS.Build_Refinery_pt(
                    "now", scv.tag, gas.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if (obs.observation.player.minerals >= 100 and
                len(scvs) > 0 and free_supply < 8):

            ccs = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
            if ccs:
                for cc in ccs:
                    cc_x, cc_y = cc.x, cc.y

                rand1, rand2 = random.randint(0, 10), random.randint(-10, 0)
                supply_depot_xy = (cc_x + rand1, cc_y + rand2) if self.base_top_left else (cc_x - rand1, cc_y - rand2)
                if 0 < supply_depot_xy[0] < 64 and 0 < supply_depot_xy[1] < 64:
                    pass
                else:
                    return actions.RAW_FUNCTIONS.no_op()

                distances = self.get_distances(obs, scvs, supply_depot_xy)
                scv = scvs[np.argmin(distances)]

                return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                    "now", scv.tag, supply_depot_xy)

        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if (len(completed_supply_depots) > 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0 and
                len(barrackses) < 3):

            brks = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)

            completed_command_center = self.get_my_completed_units_by_type(
                obs, units.Terran.CommandCenter)

            if len(barrackses) >= 1 and len(completed_command_center) == 1:
                # double commands

                commandcenters = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
                scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

                if (len(commandcenters) < 2 and obs.observation.player.minerals >= 400 and
                        len(scvs) > 0):
                    ccs_xy = (41, 21) if self.base_top_left else (17, 48)

                    distances = self.get_distances(obs, scvs, ccs_xy)
                    scv = scvs[np.argmin(distances)]

                    return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                        "now", scv.tag, ccs_xy)

            if brks:
                for brk in brks:
                    brk_x, brk_y = brk.x, brk.y

                rand1, rand2 = random.randint(1, 3), random.randint(1, 3)
                barracks_xy = (brk_x + rand1, brk_y + rand2) if self.base_top_left else (brk_x - rand1, brk_y - rand2)
                if 0 < barracks_xy[0] < 64 and 0 < barracks_xy[1] < 64:
                    pass
                else:
                    return actions.RAW_FUNCTIONS.no_op()

                distances = self.get_distances(obs, scvs, barracks_xy)
                scv = scvs[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                    "now", scv.tag, barracks_xy)

        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):

        ################# 아머리가 완성된 후 부터 토르생산 ######################
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)

        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)

        completed_armorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Armory)

        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0 and len(completed_armorys) == 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

        elif free_supply > 0 and len(completed_factorys) > 0 and len(completed_armorys) > 0:
            factory = completed_factorys[0]
            if factory.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Thor_quick("now", factory.tag)

        return actions.RAW_FUNCTIONS.no_op()

    ###############################################################################################
    ###################################### Factorys ###############################################
    ###############################################################################################

    def build_factorys(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)

        factorys = self.get_my_units_by_type(obs, units.Terran.Factory)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        ref = self.get_my_completed_units_by_type(obs, units.Terran.Refinery)
        # print("gas: ", obs.observation.player.minerals)
        # print("gas: ", obs.observation.player.gas)
        if (len(completed_barrackses) > 0 and
                obs.observation.player.minerals >= 200 and
                len(factorys) < 3 and
                len(scvs) > 0):

            if len(factorys) >= 1 and len(ref) < 4:  # 가스부족시 가스 건설
                refinerys = self.get_my_units_by_type(obs, units.Terran.Refinery)
                scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

                if (obs.observation.player.minerals >= 100 and
                        len(scvs) > 0):
                    gas = self.get_my_units_by_type(obs, units.Neutral.VespeneGeyser)[0]

                    if self.base_top_left:
                        gases = self.top_left_gas_xy
                    else:
                        gases = self.bottom_right_gas_xy

                    rc = np.random.choice([0, 1, 2, 3])
                    gas_xy = gases[rc]
                    if (gas.x, gas.y) == gas_xy:
                        distances = self.get_distances(obs, scvs, gas_xy)
                        scv = scvs[np.argmin(distances)]

                        return actions.RAW_FUNCTIONS.Build_Refinery_pt(
                            "now", scv.tag, gas.tag)

            if len(factorys) >= 1:
                rand1 = random.randint(-5, 5)
                fx, fy = factorys[0].x, factorys[0].y
                factorys_xy = (fx + rand1, fy + rand1) if self.base_top_left else (fx - rand1, fy - rand1)

            else:
                rand1, rand2 = random.randint(-2, 2), random.randint(-2, 2)  # x, y
                factorys_xy = (39 + rand1, 25 + rand2) if self.base_top_left else (17 - rand1, 40 - rand2)

            if 0 < factorys_xy[0] < 64 and 0 < factorys_xy[1] < 64 and factorys_xy != (17, 48) and factorys_xy != (
            41, 21):
                pass
            else:
                return actions.RAW_FUNCTIONS.no_op()

            distances = self.get_distances(obs, scvs, factorys_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Factory_pt(
                "now", scv.tag, factorys_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_techlab_factorys(self, obs):
        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if (len(completed_factorys) > 0 and
                obs.observation.player.minerals >= 200):

            ftrs = self.get_my_units_by_type(obs, units.Terran.Factory)

            if ftrs:
                for ftr in ftrs:
                    ftr_x, ftr_y = ftr.x, ftr.y

                factorys_xy = (ftr_x, ftr_y)
                if 0 < factorys_xy[0] < 64 and 0 < factorys_xy[1] < 64:
                    pass
                else:
                    return actions.RAW_FUNCTIONS.no_op()

                return actions.RAW_FUNCTIONS.Build_TechLab_Factory_pt(
                    "now", ftr.tag, factorys_xy)

        return actions.RAW_FUNCTIONS.no_op()

    def train_tank(self, obs):
        completed_factorytechlab = self.get_my_completed_units_by_type(
            obs, units.Terran.FactoryTechLab)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if (len(completed_factorytechlab) > 0 and obs.observation.player.minerals >= 200):

            factorys = self.get_my_units_by_type(obs, units.Terran.Factory)[0]

            if factorys.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SiegeTank_quick("now", factorys.tag)
        return actions.RAW_FUNCTIONS.no_op()

    ###############################################################################
    ############################ Build Armory ##################################

    def build_armorys(self, obs):
        completed_factory = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)

        armorys = self.get_my_units_by_type(obs, units.Terran.Armory)

        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if (len(completed_factory) > 0 and
                obs.observation.player.minerals >= 200 and
                len(armorys) < 2 and
                len(scvs) > 0):

            rand1, rand2 = random.randint(-2, 2), random.randint(-2, 2)
            armorys_xy = (36 + rand1, 20 + rand2) if self.base_top_left else (20 - rand1, 50 - rand2)
            if 0 < armorys_xy[0] < 64 and 0 < armorys_xy[1] < 64:
                pass
            else:
                return actions.RAW_FUNCTIONS.no_op()

            distances = self.get_distances(obs, scvs, armorys_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Armory_pt(
                "now", scv.tag, armorys_xy)

        elif (len(completed_factory) > 0 and
              obs.observation.player.minerals >= 200 and
              1 <= len(armorys) and
              len(scvs) > 0):
            # armory upgrade 추가
            armory = armorys[0]

            armory_xy = (armory.x, armory.y)
            # cloak_field = self.get_my_units_by_type(obs, upgrades.Upgrades.CloakingField)[0]
            if self.TerranVehicleWeaponsLevel1 == False:
                self.TerranVehicleWeaponsLevel1 = True
                return actions.RAW_FUNCTIONS.Research_TerranVehicleWeapons_quick("now", armory.tag)

            elif self.TerranVehicleWeaponsLevel1 == True and self.TerranVehicleWeaponsLevel2 == False:
                self.TerranVehicleWeaponsLevel2 = True
                return actions.RAW_FUNCTIONS.Research_TerranVehicleWeaponsLevel2_quick("now", armory.tag)

            elif self.TerranVehicleWeaponsLevel1 == True and self.TerranVehicleWeaponsLevel2 == True and self.TerranVehicleWeaponsLevel3 == False:
                self.TerranVehicleWeaponsLevel3 = True
                return actions.RAW_FUNCTIONS.Research_TerranVehicleWeaponsLevel3_quick("now", armory.tag)

        return actions.RAW_FUNCTIONS.no_op()

    ############################################################################################
    #################################### StarPort ##############################################
    def build_starports(self, obs):
        completed_factorys = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)

        starports = self.get_my_units_by_type(obs, units.Terran.Starport)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)

        if (len(completed_factorys) > 0 and
                obs.observation.player.minerals >= 200 and
                len(starports) < 1 and
                len(scvs) > 0):

            # stp_x,stp_y = (22,22), (36,46) # minerals기준 중앙부쪽 좌표

            if len(starports) >= 1:
                rand1 = random.randint(-5, 5)
                sx, sy = starports[0].x, starports[0].y
                starport_xy = (sx + rand1, sy + rand1) if self.base_top_left else (sx - rand1, sy - rand1)
            else:
                rand1, rand2 = random.randint(-5, 5), random.randint(-5, 5)
                starport_xy = (22 + rand1, 22 + rand2) if self.base_top_left else (36 - rand1, 46 - rand2)

            if 0 < starport_xy[0] < 64 and 0 < starport_xy[1] < 64:
                pass
            else:
                return actions.RAW_FUNCTIONS.no_op()

            distances = self.get_distances(obs, scvs, starport_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Starport_pt(
                "now", scv.tag, starport_xy)

        ####################### 스타포트 건설 후 팩토리 증설 #########################
        elif (len(starports) >= 1 and obs.observation.player.minerals >= 200 and
              len(completed_factorys) < 4 and len(scvs) > 0):

            if len(starports) >= 1:
                rand1 = random.randint(-5, 5)
                sx, sy = starports[0].x, starports[0].y
                factory_xy = (sx + rand1, sy + rand1) if self.base_top_left else (sx - rand1, sy - rand1)
            else:
                rand1, rand2 = random.randint(-5, 5), random.randint(-5, 5)
                factory_xy = (22 + rand1, 22 + rand2) if self.base_top_left else (36 - rand1, 46 - rand2)

            if 0 < factory_xy[0] < 64 and 0 < factory_xy[1] < 64:
                pass
            else:
                return actions.RAW_FUNCTIONS.no_op()

            distances = self.get_distances(obs, scvs, factory_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Factory_pt(
                "now", scv.tag, factory_xy)

        else:
            completed_barrackses = self.get_my_completed_units_by_type(
                obs, units.Terran.Barracks)
            marines = self.get_my_units_by_type(obs, units.Terran.Marine)

            free_supply = (obs.observation.player.food_cap -
                           obs.observation.player.food_used)

            if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                    and free_supply > 0):
                barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
                if barracks.order_length < 5:
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

        return actions.RAW_FUNCTIONS.no_op()

    def build_techlab_starports(self, obs):
        completed_starports = self.get_my_completed_units_by_type(
            obs, units.Terran.Starport)

        completed_starport_techlab = self.get_my_completed_units_by_type(
            obs, units.Terran.StarportTechLab)

        if (len(completed_starports) < 3 and
                obs.observation.player.minerals >= 200):
            stps = self.get_my_units_by_type(obs, units.Terran.Starport)

            if stps:
                for stp in stps:
                    stp_x, stp_y = stp.x, stp.y

                starport_xy = (stp_x, stp_y)

                return actions.RAW_FUNCTIONS.Build_TechLab_Starport_pt(
                    "now", stp.tag, starport_xy)

        ############ Cloak upgrade #########################
        if len(completed_starport_techlab) > 0 and self.cloaking_flag:
            # self.cloaking_flag = 0
            cloaking = upgrades.Upgrades.CloakingField

            stp_techlab = self.get_my_units_by_type(obs, units.Terran.StarportTechLab)
            if stp_techlab:
                stp_tech_xy = (stp_techlab[0].x, stp_techlab[0].y)
                cloak_field = self.get_my_units_by_type(obs, upgrades.Upgrades.CloakingField)[0]

                #                 print("stp_tech_xy: ", stp_tech_xy)
                #                 print("cloaking upgrade: ",cloak_field.tag)
                return actions.FUNCTIONS.Research_BansheeCloakingField_quick("now", cloaking)

        return actions.RAW_FUNCTIONS.no_op()

    def train_banshee(self, obs):
        completed_starporttechlab = self.get_my_completed_units_by_type(
            obs, units.Terran.StarportTechLab)

        ravens = self.get_my_units_by_type(obs, units.Terran.Raven)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if (len(completed_starporttechlab) > 0 and obs.observation.player.minerals >= 200
                and free_supply > 3):

            starports = self.get_my_units_by_type(obs, units.Terran.Starport)[0]

            ############################### cloaking detecting을 위한 Raven 생산 #######################
            if starports.order_length < 2 and len(ravens) < 3:
                return actions.RAW_FUNCTIONS.Train_Raven_quick("now", starports.tag)

            #########################################################################################

            if starports.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Banshee_quick("now", starports.tag)
        return actions.RAW_FUNCTIONS.no_op()

    ############################################################################################

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if 20 < len(marines):

            flag = random.randint(0, 2)
            if flag == 1:
                attack_xy = (38, 44) if self.base_top_left else (19, 23)
            else:
                attack_xy = (16, 45) if self.base_top_left else (42, 19)

            distances = self.get_distances(obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]
            # marine = marines

            x_offset = random.randint(-5, 5)
            y_offset = random.randint(-5, 5)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        else:
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)
            if len(barracks) > 0:
                barracks = barracks[0]
                if barracks.order_length < 5:
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)

        return actions.RAW_FUNCTIONS.no_op()

    def attack_all(self, obs):
        # 추가 유닛 생길 때 마다 추가
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        tanks = self.get_my_units_by_type(obs, units.Terran.SiegeTank)
        banshees = self.get_my_units_by_type(obs, units.Terran.Banshee)
        raven = self.get_my_units_by_type(obs, units.Terran.Raven)
        thor = self.get_my_units_by_type(obs, units.Terran.Thor)

        sieged_tanks = self.get_my_units_by_type(obs, units.Terran.SiegeTankSieged)
        total_tanks = tanks + sieged_tanks

        all_units = marines + total_tanks + banshees + raven + thor

        if 25 < len(all_units):

            flag = random.randint(0, 1000)

            if flag % 4 == 0:
                attack_xy = (39, 45) if self.base_top_left else (19, 23)
            elif flag % 4 == 1:

                attack_xy = (39, 45) if self.base_top_left else (19, 23)

                if len(tanks) > 0:
                    distances = self.get_distances(obs, tanks, attack_xy)
                    tank = tanks[np.argmax(distances)]
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    return actions.RAW_FUNCTIONS.Morph_SiegeMode_quick(
                        "now", tank.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

            elif flag % 4 == 2:
                attack_xy = (39, 45) if self.base_top_left else (19, 23)
                #### siegeMode 제거 ####
                if len(total_tanks) > 0:
                    all_tanks_tag = [tank.tag for tank in total_tanks]

                    return actions.RAW_FUNCTIONS.Morph_Unsiege_quick(
                        "now", all_tanks_tag)

            else:
                attack_xy = (17, 48) if self.base_top_left else (41, 21)

            x_offset = random.randint(-5, 5)
            y_offset = random.randint(-5, 5)

            all_tag = [unit.tag for unit in all_units]

            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", all_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        else:
            flag = random.randint(0, 1000)
            if flag % 4 == 0:
                attack_xy = (35, 25) if self.base_top_left else (25, 40)
            elif flag % 4 == 1:
                attack_xy = (35, 25) if self.base_top_left else (25, 40)

                if len(tanks) > 0:
                    tanks_tag = [tank.tag for tank in tanks]
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    return actions.RAW_FUNCTIONS.Morph_SiegeMode_quick(
                        "now", tanks_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))


            elif flag % 4 == 2:
                attack_xy = (35, 25) if self.base_top_left else (25, 40)

            else:
                attack_xy = (30, 25) if self.base_top_left else (33, 40)

            x_offset = random.randint(-1, 1)
            y_offset = random.randint(-1, 1)

            all_units = marines + banshees + raven + thor
            all_tag = [unit.tag for unit in all_units]
            if all_tag:
                return actions.RAW_FUNCTIONS.Attack_pt(
                    "now", all_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        return actions.RAW_FUNCTIONS.no_op()

    ###################################################################################
    ############################### Unit Controls #####################################

    def tank_control(self, obs):
        tanks = self.get_my_units_by_type(obs, units.Terran.SiegeTank)
        sieged_tanks = self.get_my_units_by_type(obs, units.Terran.SiegeTankSieged)

        total_tanks = tanks + sieged_tanks

        if len(total_tanks) < 8:

            if tanks:
                attack_xy = (40, 25) if self.base_top_left else (25, 40)

                distances = self.get_distances(obs, tanks, attack_xy)
                distances.sort()

                tank_tag = [t.tag for t in tanks[:4]]

                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-5, 5)
                return actions.RAW_FUNCTIONS.Morph_SiegeMode_quick(
                    "now", tank_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        else:
            #### siegeMode 제거 ####
            all_tanks_tag = [tank.tag for tank in total_tanks]
            return actions.RAW_FUNCTIONS.Morph_Unsiege_quick(
                "now", all_tanks_tag)

        return actions.RAW_FUNCTIONS.no_op()


class TerranRandomAgent(TerranAgentWithRawActsAndRawObs):
    def step(self, obs):
        super(TerranRandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)


class TerranRLAgentWithRawActsAndRawObs(TerranAgentWithRawActsAndRawObs):
    def __init__(self):
        super(TerranRLAgentWithRawActsAndRawObs, self).__init__()

        self.s_dim = 21
        self.a_dim = 19

        self.lr = 1e-4 * 1
        self.batch_size = 32
        self.gamma = 0.99
        self.memory_size = 200000
        self.eps_max = 1.0
        self.eps_min = 0.01
        self.epsilon = 1.0
        self.init_sampling = 4000
        self.target_update_interval = 10

        self.data_file_qnet = DATA_FILE_QNET
        self.data_file_qnet_target = DATA_FILE_QNET_TARGET
        self.score_file = SCORE_FILE

        self.qnetwork = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                                                  output_dim=self.a_dim,
                                                  num_neurons=[128],
                                                  hidden_act_func='ReLU',
                                                  out_act_func='Identity').to(device)

        self.qnetwork_target = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                                                         output_dim=self.a_dim,
                                                         num_neurons=[128],
                                                         hidden_act_func='ReLU',
                                                         out_act_func='Identity').to(device)

        ############################################ qnet 로드하면 이전 모델이라 학습모델 인풋 아웃풋차원이 바뀜 #########
        if os.path.isfile(self.data_file_qnet + '.pt'):
            self.qnetwork.load_state_dict(torch.load(self.data_file_qnet + '.pt'))

        if os.path.isfile(self.data_file_qnet_target + '.pt'):
            self.qnetwork_target.load_state_dict(torch.load(self.data_file_qnet_target + '.pt'))

        # initialize target network same as the main network.
        self.qnetwork_target.load_state_dict(self.qnetwork.state_dict())

        self.dqn = DQN(state_dim=self.s_dim,
                       action_dim=self.a_dim,
                       qnet=self.qnetwork,
                       qnet_target=self.qnetwork_target,
                       lr=self.lr,
                       gamma=self.gamma,
                       epsilon=self.epsilon).to(device)

        self.memory = ExperienceReplayMemory(self.memory_size)

        self.print_every = 1
        self.cum_reward = 0
        self.cum_loss = 0
        self.episode_count = 0

        self.new_game()

    def reset(self):
        super(TerranRLAgentWithRawActsAndRawObs, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.cum_reward = 0
        self.cum_loss = 0

        # epsilon scheduling
        # slowly decaying_epsilon
        self.epsilon = max(self.eps_min, self.eps_max - self.eps_min * (self.episode_count / 50))
        self.dqn.epsilon = torch.tensor(self.epsilon).to(device)

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
        if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def step(self, obs):
        super(TerranRLAgentWithRawActsAndRawObs, self).step(obs)

        # time.sleep(0.5)

        state = self.get_state(obs)
        state = torch.tensor(state).float().view(1, self.s_dim).to(device)
        action_idx = self.dqn.choose_action(state)
        action = self.actions[action_idx]
        done = True if obs.last() else False

        if self.previous_action is not None:
            experience = (self.previous_state.to(device),
                          torch.tensor(self.previous_action).view(1, 1).to(device),
                          torch.tensor(obs.reward).view(1, 1).to(device),
                          state.to(device),
                          torch.tensor(done).view(1, 1).to(device))
            self.memory.push(experience)

        self.cum_reward += obs.reward
        self.previous_state = state
        self.previous_action = action_idx

        if obs.last():
            self.episode_count = self.episode_count + 1

            if len(self.memory) >= self.init_sampling:
                # training dqn
                sampled_exps = self.memory.sample(self.batch_size)
                sampled_exps = prepare_training_inputs(sampled_exps, device)
                self.dqn.learn(*sampled_exps)

            if self.episode_count % self.target_update_interval == 0:
                self.dqn.qnet_target.load_state_dict(self.dqn.qnet.state_dict())

            if self.episode_count % self.print_every == 0:
                msg = (self.episode_count, self.cum_reward, self.epsilon)
                print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon : {:.3f}".format(*msg))

            torch.save(self.dqn.qnet.state_dict(), self.data_file_qnet + '.pt')
            torch.save(self.dqn.qnet_target.state_dict(), self.data_file_qnet_target + '.pt')

            scores_window.append(obs.reward)  # save most recent reward
            win_rate = scores_window.count(1) / len(scores_window) * 100
            tie_rate = scores_window.count(0) / len(scores_window) * 100
            lost_rate = scores_window.count(-1) / len(scores_window) * 100

            scores.append([win_rate, tie_rate, lost_rate])  # save most recent score(win_rate, tie_rate, lost_rate)
            with open(self.score_file + '.txt', "wb") as fp:
                pickle.dump(scores, fp)

            # writer.add_scalar("Loss/train", self.cum_loss/obs.observation.game_loop, self.episode_count)
            # writer.add_scalar("Score", self.cum_reward, self.episode_count)

        return getattr(self, action)(obs)

# def main(unused_argv):
#    agent1 = TerranRLAgentWithRawActsAndRawObs()
#    agent2 = TerranRandomAgent()
#    try:
#        with sc2_env.SC2Env(
#                map_name="Simple64",
#                players=[sc2_env.Agent(sc2_env.Race.terran),
#                         sc2_env.Agent(sc2_env.Race.terran)],
#                agent_interface_format=features.AgentInterfaceFormat(
#                    action_space=actions.ActionSpace.RAW,
#                    use_raw_units=True,
#                    raw_resolution=64,
#                ),
#                step_mul=8,
#                disable_fog=True,
#        ) as env:
#            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
#    except KeyboardInterrupt:
#        pass


# def main(unused_argv):
#     agent = TerranRLAgentWithRawActsAndRawObs()
#     try:
#         with sc2_env.SC2Env(
#                 map_name="Simple64",
#                 players=[sc2_env.Agent(sc2_env.Race.terran),
#                          sc2_env.Bot(sc2_env.Race.terran,
#                                      sc2_env.Difficulty.very_easy)],
#                 agent_interface_format=features.AgentInterfaceFormat(
#                     action_space=actions.ActionSpace.RAW,
#                     use_raw_units=True,
#                     raw_resolution=64,
#                 ),
#                 step_mul=8,
#                 disable_fog=True,
#         ) as env:
#             agent.setup(env.observation_spec(), env.action_spec())
#
#             timesteps = env.reset()
#             agent.reset()
#
#             while True:
#                 step_actions = [agent.step(timesteps[0])]
#                 if timesteps[0].last():
#                     break
#                 timesteps = env.step(step_actions)
#     except KeyboardInterrupt:
#         pass

# def main(unused_argv):
#     agent = TerranRLAgentWithRawActsAndRawObs()
#     try:
#         while True:
#             with sc2_env.SC2Env(
#                     map_name="Simple64",
#                     players=[sc2_env.Agent(sc2_env.Race.terran),
#                              sc2_env.Bot(sc2_env.Race.terran,
#                                          sc2_env.Difficulty.very_easy)],
#                     agent_interface_format=features.AgentInterfaceFormat(
#                         action_space=actions.ActionSpace.RAW,
#                         use_raw_units=True,
#                         raw_resolution=64,
#                     ),
#                     step_mul=8,
#                     disable_fog=True,
#                     game_steps_per_episode=0,
#                     visualize=False) as env:
#
#               agent.setup(env.observation_spec(), env.action_spec())
#
#               timesteps = env.reset()
#               agent.reset()
#
#               while True:
#                   step_actions = [agent.step(timesteps[0])]
#                   if timesteps[0].last():
#                       break
#                   timesteps = env.step(step_actions)
#
#     except KeyboardInterrupt:
#         pass


def main(unused_argv):
   agent1 = TerranRLAgentWithRawActsAndRawObs()
   try:
       with sc2_env.SC2Env(
               map_name="Simple64",
               players=[sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.terran,
                                    sc2_env.Difficulty.very_easy)],
               agent_interface_format=features.AgentInterfaceFormat(
                   action_space=actions.ActionSpace.RAW,
                   use_raw_units=True,
                   raw_resolution=64,
               ),
               step_mul=8,
               disable_fog=True,
               visualize=False
       ) as env:
           run_loop.run_loop([agent1], env, max_episodes=1000)
   except KeyboardInterrupt:
       pass

if __name__ == "__main__":
    app.run(main)
