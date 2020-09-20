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

def main(unused_argv):
   agent1 = TerranRLAgentWithRawActsAndRawObs()
   agent2 = Agent09287()
   agent3 = Agent10073()
   try:
       with sc2_env.SC2Env(
               map_name="Simple64",
               # players=[sc2_env.Agent(sc2_env.Race.terran, "08967 박석"),
               #          sc2_env.Agent(sc2_env.Race.protoss,"09287 서대웅")],
               players=[sc2_env.Agent(sc2_env.Race.protoss, "09287 서대웅"),
                        sc2_env.Agent(sc2_env.Race.protoss, "10073 오필훈")],
               agent_interface_format=features.AgentInterfaceFormat(
                   action_space=actions.ActionSpace.RAW,
                   use_raw_units=True,
                   raw_resolution=64,
               ),
               step_mul=8,
               disable_fog=True,
               visualize=False
       ) as env:
           run_loop.run_loop([agent2, agent3], env, max_episodes=1)
           env.save_replay("DRLAgentsTournamentTest02")
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