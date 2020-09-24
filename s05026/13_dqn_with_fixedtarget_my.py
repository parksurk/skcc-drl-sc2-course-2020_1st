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
from pysc2.lib import actions, features, units
from absl import app

import torch
from torch.utils.tensorboard import SummaryWriter

from skdrl.pytorch.model.mlp import NaiveMultiLayerPerceptron
from skdrl.common.memory.memory import ExperienceReplayMemory

# Network를 분리하였으므로, weight를 저정하는 공간도 2개로 분리

DATA_FILE_QNET = '13_rlagent_with_vanilla_dqn_qnet'
DATA_FILE_QNET_TARGET = '13_rlagent_with_vanilla_dqn_qnet_target'
SCORE_FILE = '13_rlagent_with_vanilla_dqn_score'

scores = []  # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('/home/jupyter/tensorboard_log')
import torch
import torch.nn as nn
import numpy as np
import random


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma  # discount future reward (미래의 보상을 조금만 반영)
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def choose_action(self, state):
        qs = self.qnet(state)
        # prob = np.random.uniform(0.0, 1.0, 1)
        # if torch.from_numpy(prob).float() <= self.epsilon:  # random
        if random.random() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    def learn(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def prepare_training_inputs(sampled_exps, device='cpu'):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.cat(states, dim=0).float().to(device)
    actions = torch.cat(actions, dim=0).to(device)
    rewards = torch.cat(rewards, dim=0).float().to(device)
    next_states = torch.cat(next_states, dim=0).float().to(device)
    dones = torch.cat(dones, dim=0).float().to(device)
    return states, actions, rewards, next_states, dones


class TerranAgentWithRawActsAndRawObs(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "train_marine",
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

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)
        return [mean_x, mean_y]

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        return [x, y]

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

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        barrack_ratio = (len(supply_depots) - len(barrackses) * 2)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (obs.observation.player.minerals >= 100 and len(supply_depots) < 25):
            free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
            exp_free_supply = (15 + len(supply_depots) * 8) - obs.observation.player.food_used
            if (exp_free_supply <= 20):
                ccs = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    if len(scvs) > 0:
                        scv = random.choice(scvs)
                        x = random.randint(mean_x - 10, mean_x + 10)
                        y = random.randint(mean_y - 10, mean_y + 10)
                        return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                            "now", scv.tag, (x, y))
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        barrack_ratio = len(supply_depots) - len(barrackses)
        if (obs.observation.player.minerals >= 150 and len(barrackses) < 15):
            if (len(supply_depots) > 0 and barrack_ratio >= 0):
                ccs = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
                if len(ccs) > 0:
                    mean_x, mean_y = self.getMeanLocation(ccs)
                    if len(scvs) > 0:
                        scv = random.choice(scvs)
                        x = random.randint(mean_x - 10, mean_x + 10)
                        y = random.randint(mean_y - 10, mean_y + 10)
                        return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, (x, y))
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100 and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-15, 15)
            y_offset = random.randint(-8, 8)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", [unit.tag for unit in marines], (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()


class TerranRandomAgent(TerranAgentWithRawActsAndRawObs):
    def step(self, obs):
        super(TerranRandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)


class TerranRLAgentWithRawActsAndRawObs(TerranAgentWithRawActsAndRawObs):
    def __init__(self):
        super(TerranRLAgentWithRawActsAndRawObs, self).__init__()

        self.s_dim = 15
        self.a_dim = 6

        self.lr = 1e-4 * 1
        self.batch_size = 32
        self.gamma = 0.99
        self.memory_size = 200000
        self.eps_max = 0.5
        self.eps_min = 0.02
        self.epsilon = 0.5
        self.init_sampling = 4000
        self.target_update_interval = 10

        self.data_file_qnet = DATA_FILE_QNET
        self.data_file_qnet_target = DATA_FILE_QNET_TARGET
        self.score_file = SCORE_FILE

        self.qnetwork = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                                                  output_dim=self.a_dim,
                                                  num_neurons=[256, 128, 64],
                                                  hidden_act_func='ReLU',
                                                  out_act_func='Identity').to(device)

        self.qnetwork_target = NaiveMultiLayerPerceptron(input_dim=self.s_dim,
                                                         output_dim=self.a_dim,
                                                         num_neurons=[256, 128, 64],
                                                         hidden_act_func='ReLU',
                                                         out_act_func='Identity').to(device)

        if os.path.isfile(self.data_file_qnet + '.pt'):
            self.qnetwork.load_state_dict(torch.load(self.data_file_qnet + '.pt', map_location=torch.device('cpu')))

        if os.path.isfile(self.data_file_qnet_target + '.pt'):
            self.qnetwork_target.load_state_dict(torch.load(self.data_file_qnet_target + '.pt', map_location=torch.device('cpu')))

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
        can_afford_marine = obs.observation.player.minerals >= 100
        too_much_minerals = obs.observation.player.minerals >= 2000
        minerals_size = round(obs.observation.player.minerals / 10, 1)

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
        enemy_Factory = self.get_enemy_units_by_type(obs, units.Terran.Factory)
        enemy_Starport = self.get_enemy_units_by_type(obs, units.Terran.Starport)
        enemy_Bunker = self.get_enemy_units_by_type(obs, units.Terran.Bunker)

        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)
        enemy_Marauder = self.get_enemy_units_by_type(obs, units.Terran.Marauder)
        enemy_Reaper = self.get_enemy_units_by_type(obs, units.Terran.Reaper)
        enemy_Hellion = self.get_enemy_units_by_type(obs, units.Terran.Hellion)
        enemy_Hellbat = self.get_enemy_units_by_type(obs, units.Terran.Hellbat)
        enemy_SiegeTank = self.get_enemy_units_by_type(obs, units.Terran.SiegeTank)
        enemy_Cyclone = self.get_enemy_units_by_type(obs, units.Terran.Cyclone)
        enemy_WidowMine = self.get_enemy_units_by_type(obs, units.Terran.WidowMine)
        enemy_Thor = self.get_enemy_units_by_type(obs, units.Terran.Thor)
        enemy_Viking = self.get_enemy_units_by_type(obs, units.Terran.VikingAssault)
        enemy_Medivac = self.get_enemy_units_by_type(obs, units.Terran.Medivac)
        enemy_Liberator = self.get_enemy_units_by_type(obs, units.Terran.Liberator)
        enemy_Raven = self.get_enemy_units_by_type(obs, units.Terran.Raven)
        enemy_Battlecruiser = self.get_enemy_units_by_type(obs, units.Terran.Battlecruiser)
        enemy_land_count = len(enemy_marines) + len(enemy_Marauder) + len(enemy_Reaper) + len(enemy_Hellion) + \
                           len(enemy_Hellbat) + len(enemy_SiegeTank) + len(enemy_Cyclone) + len(enemy_WidowMine) + len(
            enemy_Thor)
        enemy_air_count = len(enemy_Viking) + len(enemy_Medivac) + len(enemy_Medivac) + len(enemy_Liberator) + len(
            enemy_Raven) + len(enemy_Battlecruiser)
        enemy_total_count = enemy_land_count + enemy_air_count

        killed_unit_count = obs.observation.score_cumulative.killed_value_units
        killed_building_count = obs.observation.score_cumulative.killed_value_structures
        collected_minerals = obs.observation.score_cumulative.collected_minerals
        spent_minerals = obs.observation.score_cumulative.spent_minerals
        idle_worker_time = obs.observation.score_cumulative.idle_worker_time
        idle_production_time = obs.observation.score_cumulative.idle_production_time

        return (
            len(scvs),
            len(supply_depots),
            len(barrackses),
            len(marines),
            round(obs.observation.player.minerals / 10, 0),
            round(spent_minerals / 10, 0),
            idle_production_time,
            killed_unit_count,
            killed_building_count,
            len(enemy_scvs),
            len(enemy_supply_depots),
            len(enemy_barrackses),
            len(enemy_Factory),
            len(enemy_Bunker),
            enemy_total_count
        )

    def step(self, obs):
        super(TerranRLAgentWithRawActsAndRawObs, self).step(obs)
        state_org = self.get_state(obs)

        state = torch.tensor(state_org).float().view(1, self.s_dim).to(device)
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

        self.previous_state = state
        self.previous_action = action_idx
        self.cum_reward = obs.reward

        if obs.last():
            supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
            marines = self.get_my_units_by_type(obs, units.Terran.Marine)
            barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
            print("barracks : ", len(barrackses), " supply : ", len(supply_depots))
            print("marines : ", len(marines))
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


def main(unused_argv):
    agent = TerranRLAgentWithRawActsAndRawObs()
    try:
        while True:
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
                    step_mul=8,
                    disable_fog=True,
                    visualize=False) as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

                import pickle
                import numpy as np
                import matplotlib.pyplot as plt
                with open(SCORE_FILE + '.txt', "rb") as fp:
                    scores = pickle.load(fp)
                    print(np.array(scores))

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)


