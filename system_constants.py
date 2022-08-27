# System constants used across BBE

# General
NUM_OF_SIMS = 1
NUM_OF_COMPETITORS = 2
NUM_OF_EXCHANGES = 1
PRE_RACE_BETTING_PERIOD_LENGTH = 0
IN_PLAY_CUT_OFF_PERIOD = 0
SESSION_SPEED_MULTIPLIER = 50

# Data Store Attributes
RACE_DATA_FILENAME = 'data/race_event_core.csv'
#RACE_DATA_FILENAME = '/Users/freddielloyd/Documents/Uob Documents/DSP THESIS/data/race_event_core.csv'



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



#OD models

#MODEL_NAME = 'BC'
MODEL_NAME = 'fuzzy_BC'
OPINION_COMPETITOR = 1 # Bettors will be expressing opinions about this competitor. Opinions are in the range of [0,1].

MAX_OP = 1
MIN_OP = 0

# intensity of interactions
mu = 0.2 # used for all models eg. 0.2
delta = 0.25 # used for Bounded Confidence Model eg. 0.1
lmda = 0.5 # used for Relative Disagreement Model eg. 0.1

# Fuzzy membership function (triangular or trapezoidal)
FUZZY_MFX = 'triangular' 
#FUZZY_MFX = 'trapezoidal'

# Pairwise or group interactions
#INTERACTIONS = 'pairwise'
INTERACTIONS = 'group' # only for watts strogatz network within clusters

# Network Structure

#NETWORK_NAME = 'fully_connected'
NETWORK_NAME = 'watts_strogatz'
num_neighbours = 10
rewiring_prob = 0.3



# method for interaction participant selection
#interaction_method = 'direct neighbours'
#interaction_method = 'novel method'