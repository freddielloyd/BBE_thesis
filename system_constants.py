# System constants used across BBE

# General
NUM_OF_SIMS = 10
NUM_OF_COMPETITORS = 4
NUM_OF_EXCHANGES = 1
PRE_RACE_BETTING_PERIOD_LENGTH = 0
IN_PLAY_CUT_OFF_PERIOD = 0 # set as max conversation length? currently range [2,6]
SESSION_SPEED_MULTIPLIER = 1

# Data Store Attributes
RACE_DATA_FILENAME = 'data/race_event_core.csv'

# Message Protocol Numbers
EXCHANGE_UPDATE_MSG_NUM = 1
RACE_UPDATE_MSG_NUM = 2

# Exchange Attributes
MIN_ODDS = 1.1
MAX_ODDS = 20.00

# Print-Outs
TBBE_VERBOSE = False
SIM_VERBOSE = False
EXCHANGE_VERBOSE = False

# Event Attributes
# average horse races are between 5 and 12 (1005 - 2414) furlongs or could go min - max (400 - 4000)
RACE_LENGTH = 2000
MIN_RACE_LENGTH = 400
MAX_RACE_LENGTH = 4000

MIN_RACE_UNDULATION = 0
MAX_RACE_UNDULATION = 100

MIN_RACE_TEMPERATURE = 0
MAX_RACE_TEMPERATUE = 50

# Betting Agent Attributes
NUM_EX_ANTE_SIMS = 5
NUM_IN_PLAY_SIMS = 5




MAX_OP = 1
MIN_OP = 0

# intensity of interactions
mu = 0.2 # used for all models eg. 0.2
delta = 0.25 # used for Bounded Confidence Model eg. 0.1
lmda = 0.5 # used for Relative Disagreement Model eg. 0.1


#OD models
#MODEL_NAME = 'BC'
#MODEL_NAME = 'RA'
#MODEL_NAME = 'RD'
MODEL_NAME = 'fuzzy_BC'

OPINION_COMPETITOR = 1 # Bettors will be expressing opinions about this competitor. Opinions are in the range of [0,1].


# Fuzzy membership function (triangular or trapezoidal)
FUZZY_MFX = 'triangular' 
#FUZZY_MFX = 'trapezoidal' # NOT YET IMPLEMENTED FULLY

# Pairwise or group interactions
#INTERACTION_TYPE = 'pairwise'
INTERACTION_TYPE = 'group' # only for BC and fuzzy BC models

# Network Structure

#NETWORK_NAME = 'fully_connected'
NETWORK_NAME = 'watts_strogatz'

# network parameters - irrelevant for fully connected network
# shuffle agent ids before network created - creates random network vs strategies being clustered together
#SHUFFLE = 'yes'
SHUFFLE = 'no'
# num of initial neighbours before WS method rewires each one with probability rewiring_prob
NUM_NEIGHBOURS = 4
REWIRING_PROB = 0.25

# method for network interaction participant selection - irrelevant for fully connected network
INTERACTION_SELECTION = 'direct_neighbours'
#INTERACTION_SELECTION = 'across_network'


# should opinions be slightly 'muddled' depending on strength to represent ambiguity
#MUDDLE_OPINIONS = 'yes'
MUDDLE_OPINIONS = 'no'

