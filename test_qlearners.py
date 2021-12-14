# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : test_qlearners.py
#
#* Purpose :
#
#* Creation Date : 14-12-2021
#
#* Last Modified : Tuesday 14 December 2021 09:49:45 PM
#
#* Created By : Nands
#_._._._._._._._._._._._._._._._._._._._._.#

import axelrod as axl
from pprint import pprint

long_players = [s() for s in axl.rlf_strategies + axl.long_run_time_strategies]
long_tournament = axl.Tournament(long_players, seed=1)
long_results = long_tournament.play()
pprint(long_results.ranked_names)


short_players = [s() for s in axl.rlf_strategies + axl.short_run_time_strategies]
short_tournament = axl.Tournament(short_players, seed=1)
short_results = short_tournament.play()
pprint(short_results.ranked_names)
