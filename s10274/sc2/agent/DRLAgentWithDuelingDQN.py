import random
import os.path
import numpy as np

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

import torch
from torch.utils.tensorboard import SummaryWriter

from s10274.skdrl.pytorch.model.mlp import DuelingQNet
from s10274.skdrl.common.memory.memory import ExperienceReplayMemory
from s10274.skdrl.pytorch.model.dqn import DQN, prepare_training_inputs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#writer = SummaryWriter()

class TerranAgentWithRawActsAndRawObs(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "harvest_vespene",
               "build_refinery",
               "build_supply_depot",
               "build_barracks",
               "build_factory",
               "build_techlab",
               "build_command_center",
               "train_scv",
               "train_marine",
               "train_tank",
               "marine_attack",
               "tank_attack")

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
              return True

        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
              return True

        return False

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

    def step(self, obs):
        super(TerranAgentWithRawActsAndRawObs, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
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

    def harvest_vespene(self, obs):
        refineries = self.get_my_units_by_type(obs, units.Terran.Refinery)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0 and len(refineries) > 0:
            scv = random.choice(idle_scvs)
            distances = self.get_distances(obs, refineries, (scv.x, scv.y))
            refinery = refineries[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, refinery.tag)
        return actions.RAW_FUNCTIONS.no_op()

    # supply depot을 5개 건설
    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(supply_depots) < 5 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            supply_depot_coordinates = [(19, 26), (22, 26), (24, 25), (26, 24), (28, 23)] if self.base_top_left \
                else [(29, 42), (32, 42), (35, 42), (37, 42), (39, 42)]
            now_supply_depot_coordinates = [(supply_depot.x, supply_depot.y) for supply_depot in supply_depots]

            remain_supply_depot_coordinates = []
            for c in supply_depot_coordinates:
                if c not in now_supply_depot_coordinates:
                    remain_supply_depot_coordinates.append(c)

            distances = self.get_distances(obs, scvs, remain_supply_depot_coordinates[0])
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, remain_supply_depot_coordinates[0])
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_refinery(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and obs.observation.player.minerals >= 100 and len(scvs) > 0):
            vespene_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.RichVespeneGeyser,
                                   units.Neutral.VespeneGeyser
                               ]]
            scv = random.choice(scvs)
            distances = self.get_distances(obs, vespene_patches, (scv.x, scv.y))
            vespene_patch = vespene_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Refinery_pt(
                "now", scv.tag, vespene_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_factory(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        completed_refineries = self.get_my_completed_units_by_type(
            obs, units.Terran.Refinery)
        factory = self.get_my_units_by_type(obs, units.Terran.Factory)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(completed_refineries) > 0 and len(factory) == 0 and
                obs.observation.player.minerals >= 200 and obs.observation.player.vespene >= 100
                and len(scvs) > 0):
            factory_xy = (25, 21) if self.base_top_left else (30, 45)
            distances = self.get_distances(obs, scvs, factory_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Factory_pt("now", scv.tag, factory_xy)
        return actions.RAW_FUNCTIONS.no_op()

    # techlab은 factory의 addon 건물이므로 factory가 건설되어 있어야 함
    def build_techlab(self, obs):
        completed_factory = self.get_my_completed_units_by_type(
            obs, units.Terran.Factory)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_factory) > 0 and obs.observation.player.minerals >= 50 and obs.observation.player.vespene >= 50
                and len(scvs) > 0):
            return actions.RAW_FUNCTIONS.Build_TechLab_quick("now", completed_factory[0].tag)
        return actions.RAW_FUNCTIONS.no_op()

    # command center 추가 건설
    def build_command_center(self, obs):
        completed_command_center = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_command_center) > 0 and obs.observation.player.minerals >= 400
                and len(scvs) > 0):
            cc_xy = (40, 21) if self.base_top_left else (18, 47)
            distances = self.get_distances(obs, scvs, cc_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_CommandCenter_pt(
                "now", scv.tag, cc_xy)
        return actions.RAW_FUNCTIONS.no_op()

    # scv 추가 생성
    def train_scv(self, obs):
        completed_command_centers = self.get_my_completed_units_by_type(
            obs, units.Terran.CommandCenter)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        # food_workers = mineral 일꾼 + vespene 일꾼
        # print(obs.observation.player.food_workers)
        if (len(completed_command_centers) > 1 and obs.observation.player.minerals >= 50 and free_supply > 0 and
                obs.observation.player.food_workers < 20):
            command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)[1]
            if command_centers.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_centers.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def train_tank(self, obs):
        completed_factorytechlabs = self.get_my_completed_units_by_type(
            obs, units.Terran.FactoryTechLab)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        # techlab이 건설되어야 tank 생성 가능
        if (len(completed_factorytechlabs) > 0 and obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 100
                and free_supply > 0):
            factory = self.get_my_units_by_type(obs, units.Terran.Factory)[0]
            if factory.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_SiegeTank_quick("now", factory.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def marine_attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        # idle_marines = [marine for marine in marines if marine.order_length == 0]

        # rally point 설정
        if 0 < len(marines) < 10 and self.unit_type_is_selected(obs, units.Terran.Barracks):
            if self.base_top_left:
                return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 21])
            else:
                return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 46])

        # marine 10대가 모였을 때 attack
        elif len(marines) == 10:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            marins_tag = []
            for marine in marines:
                marins_tag.append(marine.tag)
            # marine 사정거리 4
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marins_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        elif len(marines) > 10:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        # # 명령을 받지 않은 marine이 10 이상 모여있으면 다른 방향으로 공격 시도
        # elif len(idle_marines) > 10:
        #     marines_tag = []
        #     for marine in idle_marines:
        #         marines_tag.append(marine.tag)
        #     attack_xy = (18, 47) if self.base_top_left else (40, 21)
        #     # marine 사정거리 4
        #     x_offset = random.randint(-4, 4)
        #     y_offset = random.randint(-4, 4)
        #     return actions.RAW_FUNCTIONS.Attack_pt(
        #         "now", marines_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        return actions.RAW_FUNCTIONS.no_op()

    def tank_attack(self, obs):
        tanks = self.get_my_units_by_type(obs, units.Terran.SiegeTank)
        # idle_tanks = [tank for tank in tanks if tank.order_length == 0]

        # rally point 설정
        if 0 < len(tanks) < 3 and self.unit_type_is_selected(obs, units.Terran.Factory):
            if self.base_top_left:
                return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 21])
            else:
                return actions.FUNCTIONS.Rally_Units_minimap("now", [29, 46])

        # tank 3대가 모였을 때 attack
        elif len(tanks) == 3:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            tanks_tag = []
            for tank in tanks:
                tanks_tag.append(tank.tag)
            # tank 사정거리 7
            x_offset = random.randint(-7, 7)
            y_offset = random.randint(-7, 7)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", tanks_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        elif len(tanks) > 3:
            attack_xy = (38, 47) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, tanks, attack_xy)
            tank = tanks[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", tank.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        # # 명령을 받지 않은 tank가 3대 이상 모여있으면 다른 방향으로 공격 시도
        # elif len(idle_tanks) > 3:
        #     tanks_tag = []
        #     for tank in idle_tanks:
        #         tanks_tag.append(tank.tag)
        #     attack_xy = (18, 47) if self.base_top_left else (40, 21)
        #     # tank 사정거리 7
        #     x_offset = random.randint(-7, 7)
        #     y_offset = random.randint(-7, 7)
        #     return actions.RAW_FUNCTIONS.Attack_pt(
        #         "now", tanks_tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))

        return actions.RAW_FUNCTIONS.no_op()

class TerranRandomAgent(TerranAgentWithRawActsAndRawObs):
    def step(self, obs):
        super(TerranRandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)

class TerranRLAgentWithRawActsAndRawObs(TerranAgentWithRawActsAndRawObs):
    def __init__(self):
        super(TerranRLAgentWithRawActsAndRawObs, self).__init__()

        self.s_dim = 21 # state의 개수
        self.a_dim = 14 # action의 개수

        self.lr = 1e-4 * 1
        self.batch_size = 32
        self.gamma = 0.99
        self.memory_size = 200000
        self.eps_max = 1.0
        self.eps_min = 0.01
        self.epsilon = 1.0
        self.init_sampling = 4000
        self.target_update_interval = 10

        self.data_file_qnet = 's10274_rlagent_with_dueling_dqn_qnet'
        self.data_file_qnet_target = 's10274_rlagent_with_dueling_dqn_qnet_target'

        self.qnetwork = DuelingQNet(input_dim=self.s_dim,
                           output_dim=self.a_dim).to(device)

        self.qnetwork_target = DuelingQNet(input_dim=self.s_dim,
                           output_dim=self.a_dim).to(device)

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

def main(unused_argv):
   agent1 = TerranRLAgentWithRawActsAndRawObs()
   try:
       with sc2_env.SC2Env(
               map_name="Simple64",
               players=[sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.terran,
                                    sc2_env.Difficulty.easy)],
               agent_interface_format=features.AgentInterfaceFormat(
                   action_space=actions.ActionSpace.RAW,
                   use_raw_units=True,
                   raw_resolution=64,
               ),
               step_mul=16,
               disable_fog=True,
               visualize=False
       ) as env:
           run_loop.run_loop([agent1], env, max_episodes=1000)
   except KeyboardInterrupt:
       pass

if __name__ == "__main__":
    app.run(main)