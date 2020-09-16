import random
import time
import math
import os.path

import numpy as np
import pandas as pd


from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

import torch
from torch.utils.tensorboard import SummaryWriter

from s09287.racepack.utils import NaiveMultiLayerPerceptron
from s09287.racepack.utils import ExperienceReplayMemory
from s09287.racepack.utils import DQN, prepare_training_inputs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#writer = SummaryWriter()

class ProtossAgentWithRawActsAndRawObs(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "harvest_vespene",
               "build_pylon",
               "build_gateway",
               "build_forge",
               "build_cyber",
               "build_assimilator",
               "build_twilight",
               "build_darkshirine",
               "build_stargate",
               "build_fleet",
               "build_robotics",
               "build_roboticsbay",
               "build_nexus",
               "train_probe",
               "train_zealot",
               "train_stalker",
               "train_adept",
               "train_dark",
               "train_voidray",
               "train_tempest",
               "train_immortal",
               "train_colossus",
               "train_mothershipcore",
               "train_mothership",
               "attack")

    def get_my_units_by_type(self, obs, unit_type):
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

    def choose_prove(self, obs):
        try:
            proves = self.get_my_units_by_type(obs, units.Protoss.Probe)
            idle_proves = [prove for prove in proves if prove.order_length == 0]
            prove = random.choice(proves) if len(idle_proves) == 0 else random.choice(idle_proves)
        except:
            prove = []
        return prove

    def step(self, obs):
        super(ProtossAgentWithRawActsAndRawObs, self).step(obs)
        if obs.first():
            nexus = self.get_my_units_by_type(
                obs, units.Protoss.Nexus)[0]
            self.base_top_left = (nexus.x < 32)
            self.gas_worker = 0

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        proves = self.get_my_units_by_type(obs, units.Protoss.Probe)
        idle_proves = [prove for prove in proves if prove.order_length == 0]
        if len(idle_proves) > 0:
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
            prove = random.choice(idle_proves)
            distances = self.get_distances(obs, mineral_patches, (prove.x, prove.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", prove.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_vespene(self, obs):
        prove = self.choose_prove(obs)
        assimilators = self.get_my_completed_units_by_type(obs, units.Protoss.Assimilator)
        if len(assimilators) >0 and len(prove) >0:
            assimilator = random.choice(assimilators)
            is_room = assimilator.ideal_harvesters - assimilator.assigned_harvesters
            if len(prove) > 0 and is_room >= 0:
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", prove.tag, assimilator.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_pylon(self, obs):
        pylons = self.get_my_units_by_type(obs, units.Protoss.Pylon)
        nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)
        gateway = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        prove = self.choose_prove(obs)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if len(nexus)+len(pylons)>0:
            if len(pylons) == 0:
                pivot = random.choice(nexus)
                pylon_pt = (pivot.x + 3, pivot.y + 3) if self.base_top_left else (pivot.x - 3, pivot.y - 3)
            else:
                pivot = random.choice(pylons+gateway)
                x_offset = random.randint(-4, 4)
                y_offset = random.randint(-4, 4)
                pylon_pt = (pivot.x + x_offset, pivot.y + y_offset)
            if obs.observation.player.minerals >= 100 and len(prove) > 0 :
                return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", prove.tag, pylon_pt)
        return actions.RAW_FUNCTIONS.no_op()

    def build_gateway(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_completed_units_by_type(obs, units.Protoss.Gateway)
        prove = self.choose_prove(obs)
        if (len(completed_pylon)+len(gateways) > 0 and obs.observation.player.minerals >= 150 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways)
            x_offset = random.randint(-3, 3)
            y_offset = random.randint(-3, 3)
            return actions.RAW_FUNCTIONS.Build_Gateway_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_robotics(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_completed_units_by_type(obs, units.Protoss.Gateway)
        cybernetics = self.get_my_completed_units_by_type(obs, units.Protoss.CyberneticsCore)
        prove = self.choose_prove(obs)
        if (len(completed_pylon) + len(gateways)> 0 and obs.observation.player.minerals >= 150 and len(prove) > 0
            and len(cybernetics) > 0):
            pivot = random.choice(completed_pylon+gateways)
            x_offset = random.randint(-3, 3)
            y_offset = random.randint(-3, 3)
            return actions.RAW_FUNCTIONS.Build_RoboticsFacility_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_roboticsbay(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_completed_units_by_type(obs, units.Protoss.Gateway)
        robotics = self.get_my_completed_units_by_type(obs, units.Protoss.RoboticsFacility)
        robotbay = self.get_my_units_by_type(obs, units.Protoss.RoboticsBay)
        prove = self.choose_prove(obs)
        if (len(completed_pylon)+len(gateways) > 0 and len(robotbay) == 0 and len(prove) > 0
            and len(robotics)>0):
            pivot = random.choice(completed_pylon + gateways)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_RoboticsBay_pt(
                "now", prove.tag, (pivot.x + x_offset, pivot.y + y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_forge(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_completed_units_by_type(obs, units.Protoss.Gateway)
        forge = self.get_my_units_by_type(obs, units.Protoss.Forge)
        prove = self.choose_prove(obs)
        if (len(completed_pylon)+len(gateways) > 0 and len(forge) == 1 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_Forge_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_cyber(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_completed_units_by_type(obs, units.Protoss.Gateway)
        cyber = self.get_my_units_by_type(obs, units.Protoss.CyberneticsCore)
        prove = self.choose_prove(obs)
        if (len(completed_pylon)+len(gateways) > 0 and len(cyber) == 0 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_CyberneticsCore_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_assimilator(self, obs):
        vespene_patches = [unit for unit in obs.observation.raw_units
                           if unit.unit_type in [
                               units.Neutral.ProtossVespeneGeyser,
                               units.Neutral.PurifierVespeneGeyser,
                               units.Neutral.RichVespeneGeyser,
                               units.Neutral.ShakurasVespeneGeyser,
                               units.Neutral.VespeneGeyser,
                           ]]
        prove = self.choose_prove(obs)
        if len(prove) > 0:
            assimilator = self.get_my_units_by_type(obs, units.Protoss.Assimilator)

            distances = self.get_distances(obs, vespene_patches, (prove.x, prove.y))
            vespene_patch = vespene_patches[np.argmin(distances)]

            if len(assimilator) < 3 and obs.observation.player.minerals >= 100:
                return actions.RAW_FUNCTIONS.Build_Assimilator_unit(
                    "now", prove.tag, vespene_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_twilight(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        twilight = self.get_my_units_by_type(obs, units.Protoss.TwilightCouncil)
        prove = self.choose_prove(obs)
        if (len(completed_pylon) +len(gateways)> 0 and len(twilight) == 0 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_TwilightCouncil_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_darkshirine(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        dark = self.get_my_units_by_type(obs, units.Protoss.DarkShrine)
        prove = self.choose_prove(obs)
        if (len(completed_pylon)+len(gateways) > 0 and len(dark) == 0 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_DarkShrine_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_stargate(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        stargate = self.get_my_units_by_type(obs, units.Protoss.Stargate)
        prove = self.choose_prove(obs)
        if (len(completed_pylon)+len(gateways)+len(stargate) > 0 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways+stargate)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_Stargate_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_fleet(self, obs):
        completed_pylon = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        stargate = self.get_my_units_by_type(obs, units.Protoss.Stargate)
        fleet = self.get_my_units_by_type(obs, units.Protoss.FleetBeacon)
        prove = self.choose_prove(obs)
        if (len(completed_pylon) > 0 and len(fleet) == 0 and len(prove) > 0):
            pivot = random.choice(completed_pylon+gateways+stargate)
            x_offset = random.randint(-2, 2)
            y_offset = random.randint(-2, 2)
            return actions.RAW_FUNCTIONS.Build_FleetBeacon_pt(
                "now", prove.tag, (pivot.x+x_offset,pivot.y+y_offset))
        return actions.RAW_FUNCTIONS.no_op()

    def build_nexus(self, obs):
        completed_nexus = self.get_my_completed_units_by_type(obs, units.Protoss.Nexus)
        if len(completed_nexus) > 0:
            nexus = completed_nexus[0]

            vespene_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.ProtossVespeneGeyser,
                                   units.Neutral.PurifierVespeneGeyser,
                                   units.Neutral.RichVespeneGeyser,
                                   units.Neutral.ShakurasVespeneGeyser,
                                   units.Neutral.VespeneGeyser,
                               ]]

            distances = self.get_distances(obs, vespene_patches, (nexus.x, nexus.y))
            ds = np.argsort(distances)
            vespene_patch = vespene_patches[ds[2]]

            if self.base_top_left:
                second_nexus_x = vespene_patch.x + 2
                second_nexus_y = vespene_patch.y + 5
            else:
                second_nexus_x = vespene_patch.x - 2
                second_nexus_y = vespene_patch.y - 5

            prove = self.choose_prove(obs)
            if (len(completed_nexus) == 1 and obs.observation.player.minerals >= 400
                and len(prove)>0):
                return actions.RAW_FUNCTIONS.Build_Nexus_pt(
                    "now", prove.tag, (second_nexus_x,second_nexus_y))
        return actions.RAW_FUNCTIONS.no_op()

    def train_probe(self, obs):
        nexus = self.get_my_units_by_type(
            obs, units.Protoss.Nexus)
        if len(nexus) > 0:
            nexus = random.choice(nexus)
            is_room = nexus.ideal_harvesters - nexus.assigned_harvesters
            if obs.observation.player.minerals >= 50 and is_room > 0:
                return actions.RAW_FUNCTIONS.Train_Probe_quick("now", nexus.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_zealot(self, obs):
        completed_gateways = self.get_my_completed_units_by_type(
            obs, units.Protoss.Gateway)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_gateways) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            gateway = random.choice(completed_gateways)
            return actions.RAW_FUNCTIONS.Train_Zealot_quick("now", gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_stalker(self, obs):
        completed_gateways = self.get_my_completed_units_by_type(
            obs, units.Protoss.Gateway)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_gateways) > 0 and obs.observation.player.minerals >= 100 and
            obs.observation.player.vespene >= 50 and free_supply > 0):
            gateway = random.choice(completed_gateways)
            return actions.RAW_FUNCTIONS.Train_Stalker_quick("now", gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_adept(self, obs):
        completed_gateways = self.get_my_completed_units_by_type(
            obs, units.Protoss.Gateway)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_gateways) > 0 and obs.observation.player.minerals >= 100 and
            obs.observation.player.vespene >= 50 and free_supply > 0):
            gateway = random.choice(completed_gateways)
            return actions.RAW_FUNCTIONS.Train_Adept_quick("now", gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_dark(self, obs):
        completed_gateways = self.get_my_completed_units_by_type(
            obs, units.Protoss.Gateway)

        completed_darkshrine = self.get_my_completed_units_by_type(
            obs, units.Protoss.DarkShrine)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_gateways)>0 and len(completed_darkshrine) > 0
                and obs.observation.player.minerals >= 125 and
            obs.observation.player.vespene >= 125 and free_supply >= 2):
            gateway = random.choice(completed_gateways)
            return actions.RAW_FUNCTIONS.Train_DarkTemplar_quick("now", gateway.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_voidray(self, obs):
        completed_stargates = self.get_my_completed_units_by_type(
            obs, units.Protoss.Stargate)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if ( obs.observation.player.minerals >= 250 and len(completed_stargates)>0 and
            obs.observation.player.vespene >= 150 and free_supply >= 4):
            stargate = random.choice(completed_stargates)
            return actions.RAW_FUNCTIONS.Train_VoidRay_quick("now", stargate.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_tempest(self, obs):
        completed_stargates = self.get_my_completed_units_by_type(
            obs, units.Protoss.Stargate)

        completed_fleetbeacon = self.get_my_completed_units_by_type(
            obs, units.Protoss.FleetBeacon)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_fleetbeacon) > 0 and obs.observation.player.minerals >= 300 and len(completed_stargates)>0 and
            obs.observation.player.vespene >= 200 and free_supply >= 4):
            stargate = random.choice(completed_stargates)
            return actions.RAW_FUNCTIONS.Train_Tempest_quick("now", stargate.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_immortal(self, obs):
        completed_robotics = self.get_my_completed_units_by_type(
            obs, units.Protoss.RoboticsFacility)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_robotics) > 0 and obs.observation.player.minerals >= 250 and
            obs.observation.player.vespene >= 100 and free_supply >= 4):
            robotics = random.choice(completed_robotics)
            return actions.RAW_FUNCTIONS.Train_Immortal_quick("now", robotics.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_colossus(self, obs):
        completed_robotics = self.get_my_completed_units_by_type(
            obs, units.Protoss.RoboticsFacility)
        completed_roboticsbay = self.get_my_completed_units_by_type(
            obs, units.Protoss.RoboticsBay)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_roboticsbay) > 0 and obs.observation.player.minerals >= 300 and len(completed_robotics)>0 and
            obs.observation.player.vespene >= 200 and free_supply >= 6):
            robotics = random.choice(completed_robotics)
            return actions.RAW_FUNCTIONS.Train_Colossus_quick("now", robotics.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_mothershipcore(self, obs):
        nexus = self.get_my_completed_units_by_type(
            obs, units.Protoss.Nexus)
        completed_cyber = self.get_my_completed_units_by_type(
            obs, units.Protoss.CyberneticsCore)
        mothershipcore = self.get_my_completed_units_by_type(
            obs, units.Protoss.MothershipCore)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_cyber) > 0 and obs.observation.player.minerals >= 100 and len(mothershipcore) == 0 and
            obs.observation.player.vespene >= 100 and free_supply >= 2 and len(nexus)>0):
            nexus = random.choice(nexus)
            return actions.RAW_FUNCTIONS.Train_MothershipCore_quick("now", nexus.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_mothership(self, obs):
        completed_fleetbeacon = self.get_my_completed_units_by_type(
            obs, units.Protoss.FleetBeacon)
        mothershipcore = self.get_my_completed_units_by_type(
            obs, units.Protoss.MothershipCore)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_fleetbeacon) > 0 and len(mothershipcore) > 0 and
                obs.observation.player.minerals >= 300 and
            obs.observation.player.vespene >= 300 and free_supply >= 6):
            mothershipcore = random.choice(mothershipcore)
            return actions.RAW_FUNCTIONS.Train_Mothership_quick("now", mothershipcore.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        zealots = self.get_my_units_by_type(obs, units.Protoss.Zealot)
        stalkers = self.get_my_units_by_type(obs, units.Protoss.Stalker)
        adepts = self.get_my_units_by_type(obs, units.Protoss.Adept)
        darks = self.get_my_units_by_type(obs, units.Protoss.DarkTemplar)
        tempests = self.get_my_units_by_type(obs, units.Protoss.Tempest)
        voidrays = self.get_my_units_by_type(obs, units.Protoss.VoidRay)
        immortals = self.get_my_units_by_type(obs, units.Protoss.Immortal)
        colossus = self.get_my_units_by_type(obs, units.Protoss.Colossus)
        mothercores = self.get_my_units_by_type(obs, units.Protoss.MothershipCore)
        motherships = self.get_my_units_by_type(obs, units.Protoss.Mothership)

        army = zealots+stalkers+adepts+darks+tempests+mothercores+motherships+voidrays+immortals+colossus
        if len(army) > 6:
            armytags = [a.tag for a in army]
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, army, attack_xy)
            # armed = army[np.argmax(distances)]
            x_offset = random.randint(-20, 20)
            y_offset = random.randint(-20, 20)
            target_x = attack_xy[0] + x_offset if (attack_xy[0] + x_offset) > 0 else 0
            target_y = attack_xy[1] + y_offset if (attack_xy[1] + y_offset) > 0 else 0
            if target_x >= 0 and target_y >= 0:
                return actions.RAW_FUNCTIONS.Attack_pt(
                    "now", armytags, (target_x, target_y))
        return actions.RAW_FUNCTIONS.no_op()

class ProtossRandomAgent(ProtossAgentWithRawActsAndRawObs):
    def step(self, obs):
        super(ProtossRandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)

class ProtossRLAgentWithRawActsAndRawObs(ProtossAgentWithRawActsAndRawObs):
    def __init__(self):
        super(ProtossRLAgentWithRawActsAndRawObs, self).__init__()

        self.s_dim = 35
        self.a_dim = 27

        self.lr = 1e-4 * 1
        self.batch_size = 32
        self.gamma = 0.99
        self.memory_size = 200000
        self.eps_max = 1.0
        self.eps_min = 0.01
        self.epsilon = 1.0
        self.init_sampling = 4000
        # self.init_sampling = 1000
        self.target_update_interval = 10

        self.data_file_qnet = 's09287_rlagent_with_vanilla_dqn_qnet'
        self.data_file_qnet_target = 's09287_rlagent_with_vanilla_dqn_qnet_target'

        self.qnetwork = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                           output_dim=self.a_dim,
                           num_neurons=[256],
                           hidden_act_func='ReLU',
                           out_act_func='Identity').to(device)

        self.qnetwork_target = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                           output_dim=self.a_dim,
                           num_neurons=[256],
                           hidden_act_func='ReLU',
                           out_act_func='Identity').to(device)

        if os.path.isfile(self.data_file_qnet + '.pt'):
            self.qnetwork.load_state_dict(torch.load(self.data_file_qnet + '.pt', map_location=device))

        if os.path.isfile(self.data_file_qnet_target + '.pt'):
            self.qnetwork_target.load_state_dict(torch.load(self.data_file_qnet_target + '.pt', map_location=device))

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
        super(ProtossRLAgentWithRawActsAndRawObs, self).reset()
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
        probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
        idle_probes = [probe for probe in probes if probe.order_length == 0]
        nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)
        pylons = self.get_my_units_by_type(obs, units.Protoss.Pylon)
        stargates = self.get_my_units_by_type(
            obs, units.Protoss.Stargate)
        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        robotics = self.get_my_units_by_type(
            obs, units.Protoss.RoboticsFacility)
        zealots = self.get_my_units_by_type(obs, units.Protoss.Zealot)
        stalkers = self.get_my_units_by_type(obs, units.Protoss.Stalker)
        adepts = self.get_my_units_by_type(obs, units.Protoss.Adept)
        darks = self.get_my_units_by_type(obs, units.Protoss.DarkTemplar)
        tempests = self.get_my_units_by_type(obs, units.Protoss.Tempest)
        voidrays = self.get_my_units_by_type(obs, units.Protoss.VoidRay)
        immortals = self.get_my_units_by_type(obs, units.Protoss.Immortal)
        colossus = self.get_my_units_by_type(obs, units.Protoss.Colossus)
        mothercores = self.get_my_units_by_type(obs, units.Protoss.MothershipCore)
        motherships = self.get_my_units_by_type(obs, units.Protoss.Mothership)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_gateways = obs.observation.player.minerals >= 150
        can_afford_zealot = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_mules = self.get_enemy_units_by_type(obs, units.Terran.MULE)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_factories = self.get_enemy_units_by_type(
            obs, units.Terran.Factory)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_starport = self.get_enemy_units_by_type(
            obs, units.Terran.Starport)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
        enemy_marauder = self.get_enemy_units_by_type(obs, units.Terran.Marauder)
        enemy_hellion = self.get_enemy_units_by_type(obs, units.Terran.Hellion)
        enemy_tank = self.get_enemy_units_by_type(obs, units.Terran.SiegeTank)
        enemy_viking = self.get_enemy_units_by_type(obs, units.Terran.VikingFighter)
        enemy_medivac = self.get_enemy_units_by_type(obs, units.Terran.Medivac)
        enemy_banshee = self.get_enemy_units_by_type(obs, units.Terran.Banshee)

        return (len(nexus),
                len(probes),
                len(idle_probes),
                len(pylons),
                len(gateways),
                len(stargates),
                len(robotics),
                len(zealots),
                len(stalkers),
                len(adepts),
                len(darks),
                len(tempests),
                len(voidrays),
                len(immortals),
                len(colossus),
                len(mothercores+motherships),
                free_supply,
                can_afford_supply_depot,
                can_afford_gateways,
                can_afford_zealot,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_mules),
                len(enemy_supply_depots),
                len(enemy_factories),
                len(enemy_barrackses),
                len(enemy_starport),
                len(enemy_marines),
                len(enemy_marauder),
                len(enemy_hellion),
                len(enemy_tank),
                len(enemy_viking),
                len(enemy_medivac),
                len(enemy_banshee)
                )

    def step(self, obs):
        super(ProtossRLAgentWithRawActsAndRawObs, self).step(obs)

        #time.sleep(0.5)

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

            #writer.add_scalar("Loss/train", self.cum_loss/obs.observation.game_loop, self.episode_count)
            #writer.add_scalar("Score", self.cum_reward, self.episode_count)

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
   agent1 = ProtossRLAgentWithRawActsAndRawObs()
   try:
       with sc2_env.SC2Env(
               map_name="Simple64",
               players=[sc2_env.Agent(sc2_env.Race.protoss),
                        sc2_env.Bot(sc2_env.Race.terran,
                                    sc2_env.Difficulty.hard)],
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
