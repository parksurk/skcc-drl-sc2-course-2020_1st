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


from baseline.sc2.agent.DRLAgentWithVanillaDQN import TerranRLAgentWithRawActsAndRawObs
from s10073.sc2.agent.DRLAgentWithVanillaDQN_phil import ProtossRLAgentWithRawActsAndRawObs as Agent10073
from s09287.ProtossDQN import ProtossRLAgentWithRawActsAndRawObs as Agent09287
from s09360.sc2.agent.DRLAgentWithVanillaDQN import TerranRLAgentWithRawActsAndRawObs as Agent09360
from s10472.sc2.agent.RLAgent import ZergAgent as Agent10472
from s10336.sc2.agent.DRLAgentWithVanillaDQN import TerranRLAgentWithRawActsAndRawObs as Agent10336
from s10071.sc2.agent.DRLAgentWithVDQN_mod_final import TerranRLAgentWithRawActsAndRawObs as Agent10071
from s10395.sc2.agent.protoss_DRLAgentWithVanillaDQN import ProtossRLAgentWithRawActsAndRawObs as Agent10395
from s10274.sc2.agent.DRLAgentWithDuelingDQN import TerranRLAgentWithRawActsAndRawObs as Agent10274

def main(unused_argv):
   agent_baseline = TerranRLAgentWithRawActsAndRawObs()
   T_09360 = Agent09360() # sc2_env.Race.terran, "09360 조용준"
   Z_10472 = Agent10472() # sc2_env.Race.zerg, "10472 오수은"
   P_09287 = Agent09287() # sc2_env.Race.protoss, "09287 서대웅"
   T_10336 = Agent10336() # sc2_env.Race.terran, "10336 김명환"
   T_10071 = Agent10071() # sc2_env.Race.terran, "10071 오동훈"
   P_10395 = Agent10395() # sc2_env.Race.protoss, "10395 이현호"
   P_10073 = Agent10073() # sc2_env.Race.protoss, "10073 오필훈"
   T_10274 = Agent10071() # sc2_env.Race.terran, "10274 최지은"

   try:
       with sc2_env.SC2Env(
               map_name="Simple64",
               # players=[sc2_env.Agent(sc2_env.Race.terran, "09360 조용준"),
               #          sc2_env.Agent(sc2_env.Race.zerg, "10472 오수은")],
               players=[sc2_env.Agent(sc2_env.Race.protoss, "09287 서대웅"),
                        sc2_env.Agent(sc2_env.Race.terran, "10336 김명환")],
               # players=[sc2_env.Agent(sc2_env.Race.terran, "10071 오동훈"),
               #          sc2_env.Agent(sc2_env.Race.protoss, "10395 이현호")],
               # players=[sc2_env.Agent(sc2_env.Race.protoss, "10073 오필훈"),
               #          sc2_env.Agent(sc2_env.Race.terran, "10274 최지은")],
               agent_interface_format=features.AgentInterfaceFormat(
                   action_space=actions.ActionSpace.RAW,
                   use_feature_units=True,
                   feature_dimensions=features.Dimensions(screen=32, minimap=32),
                   use_raw_units=True,
                   use_raw_actions=True,
                   raw_resolution=64,
               ),
               step_mul=8,
               disable_fog=True,
               visualize=False
       ) as env:
           run_loop.run_loop([P_09287, T_10336], env, max_episodes=1)
           env.save_replay("DRLAgentsTournament_ROUND1")
   except KeyboardInterrupt:
       pass


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


# def main(unused_argv):
#    agent1 = TerranRLAgentWithRawActsAndRawObs()
#    try:
#        with sc2_env.SC2Env(
#                map_name="Simple64",
#                players=[sc2_env.Agent(sc2_env.Race.terran),
#                         sc2_env.Bot(sc2_env.Race.terran,
#                                     sc2_env.Difficulty.very_easy)],
#                agent_interface_format=features.AgentInterfaceFormat(
#                    action_space=actions.ActionSpace.RAW,
#                    use_raw_units=True,
#                    raw_resolution=64,
#                ),
#                step_mul=8,
#                disable_fog=True,
#                visualize=False
#        ) as env:
#            run_loop.run_loop([agent1], env, max_episodes=1)
#    except KeyboardInterrupt:
#        pass
#


# def main(unused_argv):
#    agent1 = ProtossRLAgentWithRawActsAndRawObs()
#    try:
#        with sc2_env.SC2Env(
#                map_name="Simple64",
#                players=[sc2_env.Agent(sc2_env.Race.protoss),
#                         sc2_env.Bot(sc2_env.Race.terran,
#                                     sc2_env.Difficulty.very_easy)],
#                agent_interface_format=features.AgentInterfaceFormat(
#                    action_space=actions.ActionSpace.RAW,
#                    use_raw_units=True,
#                    raw_resolution=64,
#                ),
#                step_mul=8,
#                disable_fog=True,
#                visualize=False
#        ) as env:
#            run_loop.run_loop([agent1], env, max_episodes=1)
#            env.save_replay("DRLAgentsTournamentTest")
#    except KeyboardInterrupt:
#        pass

if __name__ == "__main__":
    app.run(main)
