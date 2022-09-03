### ~ THREADED BRISTOL BETTING EXCHANGE ~ ###


import math, threading, time, queue, random, csv, pandas

import os
import sys

sys.path
file_dir = os.path.dirname("/Users/freddielloyd/Documents/Uob Documents/DSP THESIS/Git_Repo/")
sys.path.append(file_dir)
sys.path

import config
from copy import deepcopy


from system_constants import *
from betting_agents import *
import numpy as np
from race_simulator import Simulator
from ex_ante_odds_generator import *
from exchange import Exchange
from message_protocols import *
from session_stats import *
from ODmodels import *

from plotting import plotting_main

import os

os.chdir('/Users/freddielloyd/Documents/Uob Documents/DSP THESIS')

class Session:

    def __init__(self):
        # Initialise exchanges
        self.exchanges = {}
        self.exchangeOrderQs = {}
        self.exchangeThreads = []

        # Initialise betting agents
        self.bettingAgents = {}
        self.bettingAgentQs = {}
        self.bettingAgentThreads = []

        self.OpinionDynamicsPlatform = None

        # Needed attributes
        self.startTime = None
        self.numberOfTimesteps = None
        self.lengthOfRace = None
        self.event = threading.Event()
        self.endOfInPlayBettingPeriod = None
        self.winningCompetitor = None
        self.distances = None

        # Record keeping attributes
        self.tape = []
        self.priceRecord = {}
        self.spreads = {}
        self.opinion_hist = {'id': [], 'time': [], 'opinion': [], 'competitor': []}
        self.opinion_hist_l = {'id': [], 'type': [], 'time': [], 'opinion': [], 'competitor': []}
        self.opinion_hist_e = {'id': [], 'time': [], 'opinion': [], 'competitor': []}
        self.opinion_hist_g = {'id': [], 'time': [], 'opinion': [], 'competitor': []}
        self.opinion_hist_s = {'id': [], 'time': [], 'opinion': [], 'competitor': []}
        self.competitor_odds = {'time': [], 'odds': [], 'competitor': []}
        self.competitor_distances = {'time': [], 'distance': [], 'competitor': []}
        
        self.pairwise_interaction_log = {'type': [], 
                                         'time': [], 
                                         'length': [],
                                         'bettor1': [], 
                                         'bettor1_id': [],
                                         'b1_local_op': [], 
                                         'bettor2': [],
                                         'bettor2_id': [],
                                         'deg_of_connection': [],
                                         'b2_local_op': [],
                                         'b2_expressed_op': [],
                                         'local_op_gap': [],
                                         'weight': [],
                                         'b1_new_local_op': [],
                                         'b1_op_change': [],
                                         'b2_new_local_op': []}     
        
        self.group_interaction_log = {'type': [], 
                                      'conv_id': [], 
                                      'time': [], 
                                      'length': [],
                                      'bettor1': [], 
                                      'bettor1_id': [],
                                      'b1_local_op': [], 
                                      'num_bettors': [],
                                      'bettors': [],
                                      'bettors_ids': [],
                                      'degs_of_connection': [],
                                      'bettors_local_ops': [],
                                      'bettors_expressed_ops': [],
                                      #'local_op_gap': [],
                                      'weights': [],
                                      'ops_x_weights': [],
                                      'b1_new_local_op': [],
                                      'b1_op_change': []}
        
        self.interaction_logs = {'pairwise' : self.pairwise_interaction_log,
                                 'group' : self.group_interaction_log}

        self.generateRaceData()
        self.initialiseThreads()
        
        

    def exchangeLogic(self, exchange, exchangeOrderQ):
        """
        Logic for thread running the exchange
        """
        print("EXCHANGE " + str(exchange.id) + " INITIALISED...")
        self.event.wait()
        # While event is running, run logic for exchange

        competitor_odds = {'time': [], 'odds': [], 'competitor': []}

        while self.event.isSet():
            timeInEvent = (time.time() - self.startTime) / SESSION_SPEED_MULTIPLIER
            try: order = exchangeOrderQ.get(block=False)
            except: continue

            marketUpdates = {}
            for i in range(NUM_OF_EXCHANGES):
                marketUpdates[i] = self.exchanges[i].publishMarketState(timeInEvent)


            if timeInEvent < self.endOfInPlayBettingPeriod:
                self.OpinionDynamicsPlatform.initiate_conversations(timeInEvent)
                self.OpinionDynamicsPlatform.update_opinions(timeInEvent, marketUpdates)

            else:
                self.OpinionDynamicsPlatform.settle_opinions(self.winningCompetitor)

            (transactions, markets) = exchange.processOrder(timeInEvent, order)

            if transactions != None:
                for id, q in self.bettingAgentQs.items():
                    update = exchangeUpdate(transactions, order, markets)
                    q.put(update)


    def agentLogic(self, agent, agentQ):
        """
        Logic for betting agent threads
        """
        print("AGENT " + str(agent.shuffled_id) + " INITIALISED...")
        #print(agent)
        # Need to have pre-event betting period
        self.event.wait()
        # Whole event is running, run logic for betting agents
        while self.event.isSet(): 
            time.sleep(0.01)
            timeInEvent = (time.time() - self.startTime) / SESSION_SPEED_MULTIPLIER
            order = None
            trade = None


            while agentQ.empty() is False:
                qItem = agentQ.get(block = False)
                if qItem.protocolNum == EXCHANGE_UPDATE_MSG_NUM:
                    for transaction in qItem.transactions:
                        if transaction['backer'] == agent.id: agent.bookkeep(transaction, 'Backer', qItem.order, timeInEvent)
                        if transaction['layer'] == agent.id: agent.bookkeep(transaction, 'Layer', qItem.order, timeInEvent)

                elif qItem.protocolNum == RACE_UPDATE_MSG_NUM:
                    agent.observeRaceState(qItem.timestep, qItem.compDistances)
                else:
                    print("INVALID MESSAGE")



            marketUpdates = {}
            for i in range(NUM_OF_EXCHANGES):
                marketUpdates[i] = self.exchanges[i].publishMarketState(timeInEvent)

            agent.respond(timeInEvent, marketUpdates, trade)
            order = agent.getorder(timeInEvent, marketUpdates)
            
            


            if agent.id == 0:
                for i in range(NUM_OF_COMPETITORS):
                    self.competitor_odds['time'].append(timeInEvent)
                    self.competitor_odds['competitor'].append(i)
                    if marketUpdates[0][i]['backs']['n'] > 0:
                        self.competitor_odds['odds'].append(marketUpdates[0][i]['backs']['best'])
                    else:
                        self.competitor_odds['odds'].append(marketUpdates[0][i]['backs']['worst'])

                    self.competitor_distances['competitor'].append(i)
                    self.competitor_distances['time'].append(timeInEvent)
                    if len(agent.currentRaceState) == 0:
                        self.competitor_distances['distance'].append(0)
                    else:
                        self.competitor_distances['distance'].append(agent.currentRaceState[i])


            #self.opinion_hist['id'].append(agent.id)
            self.opinion_hist['id'].append(agent.shuffled_id)
            self.opinion_hist['time'].append(timeInEvent)
            self.opinion_hist['opinion'].append(agent.opinion)
            self.opinion_hist['competitor'].append(OPINION_COMPETITOR)

            #self.opinion_hist_e['id'].append(agent.id)
            self.opinion_hist_e['id'].append(agent.shuffled_id)
            self.opinion_hist_e['time'].append(timeInEvent)
            self.opinion_hist_e['opinion'].append(agent.event_opinion)
            self.opinion_hist_e['competitor'].append(OPINION_COMPETITOR)

            #self.opinion_hist_l['id'].append(agent.id)
            self.opinion_hist_l['id'].append(agent.shuffled_id)
            self.opinion_hist_l['type'].append(agent.name)
            self.opinion_hist_l['time'].append(timeInEvent)
            self.opinion_hist_l['opinion'].append(agent.local_opinion)
            self.opinion_hist_l['competitor'].append(OPINION_COMPETITOR)

            #self.opinion_hist_g['id'].append(agent.id)
            self.opinion_hist_g['id'].append(agent.shuffled_id)
            self.opinion_hist_g['time'].append(timeInEvent)
            self.opinion_hist_g['opinion'].append(agent.global_opinion)
            self.opinion_hist_g['competitor'].append(OPINION_COMPETITOR)

            #self.opinion_hist_s['id'].append(agent.id)
            self.opinion_hist_s['id'].append(agent.shuffled_id)
            self.opinion_hist_s['time'].append(timeInEvent)
            self.opinion_hist_s['opinion'].append(agent.strategy_opinion)
            self.opinion_hist_s['competitor'].append(OPINION_COMPETITOR)

            if order != None:

                if TBBE_VERBOSE:
                    print(order)
                agent.numOfBets = agent.numOfBets + 1
                self.exchangeOrderQs[order.exchange].put(order)


        print("ENDING AGENT " + str(agent.shuffled_id))
        return 0

    def populateMarket(self):
        """
        Populate market with betting agents as specified in config file
        """
        #def initAgent(name, quantity, id):
        def initAgent(name, id):

            uncertainty = 1.0

            local_opinion = 1/ NUM_OF_COMPETITORS

            #
            # if name == 'Test': return Agent_Test(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Random': return Agent_Random(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Leader_Wins': return Agent_Leader_Wins(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Underdog': return Agent_Underdog(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Back_Favourite': return Agent_Back_Favourite(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Linex': return Agent_Linex(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Arbitrage': return Agent_Arbitrage(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Arbitrage2': return Agent_Arbitrage2(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)
            # if name == 'Priveledged': return Agent_Priveledged(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod)

            if name == 'Agent_Opinionated_Random': return Agent_Opinionated_Random(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP )
            if name == 'Agent_Opinionated_Leader_Wins': return Agent_Opinionated_Leader_Wins(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP )
            if name == 'Agent_Opinionated_Underdog': return Agent_Opinionated_Underdog(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP)
            if name == "Agent_Opinionated_Back_Favourite": return Agent_Opinionated_Back_Favourite(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP)
            if name == 'Agent_Opinionated_Linex': return Agent_Opinionated_Linex(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod, 0, local_opinion,uncertainty, MIN_OP, MAX_OP)

            if name == 'Agent_Opinionated_Priviledged': return Agent_Opinionated_Priviledged(id, name, self.lengthOfRace, self.endOfInPlayBettingPeriod, 1, local_opinion, uncertainty, MIN_OP, MAX_OP)


            #if name == 'Agent_Opinionated_Random': return Agent_Opinionated_Random(id, name, lengthOfRace, endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP )
            #if name == 'Agent_Opinionated_Leader_Wins': return Agent_Opinionated_Leader_Wins(id, name, lengthOfRace, endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP )
            #if name == 'Agent_Opinionated_Underdog': return Agent_Opinionated_Underdog(id, name, lengthOfRace, endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP)
            #if name == "Agent_Opinionated_Back_Favourite": return Agent_Opinionated_Back_Favourite(id, name, lengthOfRace, endOfInPlayBettingPeriod, 0, local_opinion, uncertainty, MIN_OP, MAX_OP)
            #if name == 'Agent_Opinionated_Linex': return Agent_Opinionated_Linex(id, name, lengthOfRace, endOfInPlayBettingPeriod, 0, local_opinion,uncertainty, MIN_OP, MAX_OP)

            #if name == 'Agent_Opinionated_Priviledged': return Agent_Opinionated_Priviledged(id, name, lengthOfRace, endOfInPlayBettingPeriod, 1, local_opinion, uncertainty, MIN_OP, MAX_OP)

        # shuffle agents to prevent agents of same type automatically having adjacent ids
        # and therefore connected in network
        id = 0
        for agent in config.agents:
            type = agent[0]
            for i in range(agent[1]):
                #self.bettingAgents[id] = initAgent(agent[0], agent[1], id)
                self.bettingAgents[id] = initAgent(agent[0], id)
                id = id + 1

        to_shuffle = list(self.bettingAgents.values()) # cant shuffle dict so need to make list then convert back after shuffling
        #print(to_shuffle)
        random.shuffle(to_shuffle)  # ensures all types of bettors are mixed up rather than all adjacent
        #print(to_shuffle)
        
        self.bettingAgents = dict(zip(self.bettingAgents, to_shuffle))
        #print(self.bettingAgents)
        
        for i in range(len(self.bettingAgents.values())):
            agent = list(self.bettingAgents.values())[i]
            #print(agent, agent.id)
            new_id = list(self.bettingAgents.keys())[i]
            agent.shuffled_id = new_id # set id to new id after shuffling so can match up as needed
            #print(agent, agent.id, agent.shuffled_id)


    def initialiseExchanges(self):
        """
        Initialise exchanges, returns list of exchange objects
        """
        for i in range(NUM_OF_EXCHANGES):
            self.exchanges[i] = Exchange(i, NUM_OF_COMPETITORS) # NUM_OF_COMPETITORS may be changed to list of competitor objects that are participating
            self.exchangeOrderQs[i] = queue.Queue()

    def initialiseBettingAgents(self):
        """
        Initialise betting agents
        """
        self.populateMarket()
        self.OpinionDynamicsPlatform = OpinionDynamicsPlatform(list(self.bettingAgents.values()), 
                                                               MODEL_NAME, 
                                                               NETWORK_NAME, 
                                                               INTERACTION_TYPE,
                                                               INTERACTION_SELECTION,
                                                               self.interaction_logs,
                                                               MUDDLE_OPINIONS)
        
        # Create threads for all betting agents that wait until event session
        # has started
        for id, agent in self.bettingAgents.items():
            #print(id, agent)
            self.bettingAgentQs[id] = queue.Queue()
            thread = threading.Thread(target = self.agentLogic, args = [agent, self.bettingAgentQs[id]])
            self.bettingAgentThreads.append(thread)


    def updateRaceQ(self, timestep):
        """
        Read in race data and update agent queues with competitor distances at timestep
        """
        with open(RACE_DATA_FILENAME, 'r') as file:
            reader = csv.reader(file)
            r = [row for index, row in enumerate(reader) if index == timestep]
        time = r[0][0]
        compDistances = {}
        for c in range(NUM_OF_COMPETITORS):
            compDistances[c] = float(r[0][c+1])

        # Create update
        update = raceUpdate(time, compDistances)

        for id, q in self.bettingAgentQs.items():
            q.put(update)

    def preRaceBetPeriod(self):
        print("Start of pre-race betting period, lasting " + str(PRE_RACE_BETTING_PERIOD_LENGTH))
        time.sleep(PRE_RACE_BETTING_PERIOD_LENGTH / SESSION_SPEED_MULTIPLIER)
        print("End of pre-race betting period")
        # marketUpdates = {}
        # for id, ex in exchanges.items():
        #     timeInEvent = time.time() - startTime
        #     print("Exchange " + str(id) + " markets: ")
        #     print(exchanges[id].publishMarketState(timeInEvent))


    def eventSession(self, simulationId):
        """
        Set up and management of race event
        """

        # Record start time
        self.startTime = time.time()

        # Start exchange threads
        for id, exchange in self.exchanges.items():
            thread = threading.Thread(target = self.exchangeLogic, args = [exchange, self.exchangeOrderQs[id]])
            self.exchangeThreads.append(thread)
        
        for thread in self.exchangeThreads:
            thread.start()

        # Start betting agent threads
        for thread in self.bettingAgentThreads:
            thread.start()


        # Initialise event
        self.event.set()

        time.sleep(0.01)

        # Pre-race betting period
        self.preRaceBetPeriod()


        # have loop which runs until competitor has won race
        i = 0
        while(i < self.numberOfTimesteps):
            self.updateRaceQ(i+1)
            i = i+1
            if TBBE_VERBOSE: print(i)
            print(i)
            time.sleep(1 / SESSION_SPEED_MULTIPLIER)

        # End event
        self.event.clear()

        # Close threads 
        for thread in self.exchangeThreads: thread.join() # THIS IS the problem
        for thread in self.bettingAgentThreads: thread.join()
        

        print("Simulation complete")

        print("Writing data....")
        for id, ex in self.exchanges.items():
            for orderbook in ex.compOrderbooks:
                for trade in orderbook.tape:
                    #print(trade)
                    self.tape.append(trade)

        # Settle up all transactions over all exchanges
        for id, ex in self.exchanges.items():
            ex.settleUp(self.bettingAgents, self.winningCompetitor)

        # for id, exchange in exchanges.items():
        #     exchange.tapeDump('transactions.csv', 'a', 'keep')

        #for id, agent in self.bettingAgents.items():
        #    print("Agent " + str(id) + "\'s final balance: " + str(agent.balance))

        createstats(self.bettingAgents, simulationId, self.tape, self.priceRecord, self.spreads)

    def initialiseThreads(self):
        self.initialiseExchanges()
        self.initialiseBettingAgents()

    def generateRaceData(self):
        # Create race event data
        race = Simulator(NUM_OF_COMPETITORS)

        compPool = deepcopy(race.competitors)
        raceAttributes = deepcopy(race.race_attributes)
        
        race.printInitialConditions()
        race.printCompPool()

        # create simulations for procurement of ex-ante odds for priveledged betters
        createExAnteOdds(compPool, raceAttributes)

        race.run("core")
        
        self.numberOfTimesteps = race.numberOfTimesteps
        self.lengthOfRace = race.race_attributes.length
        self.winningCompetitor = race.winner
        self.distances = race.raceData
        print(race.winningTimestep)
        self.endOfInPlayBettingPeriod = race.winningTimestep - IN_PLAY_CUT_OFF_PERIOD
        print(self.endOfInPlayBettingPeriod)


        createInPlayOdds(self.numberOfTimesteps)



class BBE(Session):
    def __init__(self, seed):
        self.session = None
        self.seed = seed
        
        #random.seed(1) # if set here race conditions and competitor pool different
        return


    # MAIN LOOP
    # argFuncf is an optional function which sets up a new session (takes in a session)
    def runSession(self, argFunc=None):
        # Simulation attributes
        currentSimulation = 0
        ####################

        # set things up
        # have while loop for running multiple races
        # within loop instantiate competitors into list
        # run simulation and matching engine
        while currentSimulation < NUM_OF_SIMS:
            
            sim_start_time = time.time()
            
            #random.seed(1) # if set here race and transactions identical on every sim
            #np.random.seed(26) # no np randoms used

            
            simulationId = str(currentSimulation)
            print("Simulation: " + simulationId)
            # Start up thread for race on which all other threads will wait
            
            random.seed(self.seed)  #set here before Session initialised for same race
            self.session = Session()
            
            if argFunc:
                argFunc(self.session)
            self.session.eventSession(currentSimulation)
            
            plotting_main(self.seed) # produce desired plots from plotting.py on each sim

            #currentSimulation = currentSimulation + 1

            # Opinion Dynamics results:
    
            opinion_hist_df = pandas.DataFrame.from_dict(self.session.opinion_hist)
            opinion_hist_df.to_csv('data/opinions/opinions{}.csv'.format(currentSimulation), index=False)
    
            opinion_hist_df_l = pandas.DataFrame.from_dict(self.session.opinion_hist_l)
            opinion_hist_df_l.to_csv('data/opinions/local_opinions{}.csv'.format(currentSimulation), index=False)
    
            opinion_hist_df_g = pandas.DataFrame.from_dict(self.session.opinion_hist_g)
            opinion_hist_df_g.to_csv('data/opinions/global_opinions{}.csv'.format(currentSimulation), index=False)
    
            opinion_hist_df_e = pandas.DataFrame.from_dict(self.session.opinion_hist_e)
            opinion_hist_df_e.to_csv('data/opinions/event_opinions{}.csv'.format(currentSimulation), index=False)
    
            competitor_odds_df = pandas.DataFrame.from_dict(self.session.competitor_odds)
            competitor_odds_df.to_csv('data/competitor_odds{}.csv'.format(currentSimulation), index=False)
    
            competitor_distances_df = pandas.DataFrame.from_dict(self.session.competitor_distances)
            competitor_distances_df.to_csv('data/competitor_distances{}.csv'.format(currentSimulation), index=False)
    
            opinion_hist_s_df = pandas.DataFrame.from_dict(self.session.opinion_hist_s)
            opinion_hist_s_df.to_csv('data/opinions/opinion_hist_s{}.csv'.format(currentSimulation), index=False)
            
            
            if INTERACTION_TYPE == 'pairwise':
                interaction_log_df = pandas.DataFrame.from_dict(self.session.interaction_logs['pairwise'], orient='index')
                interaction_log_df = interaction_log_df.transpose() # essential with orient = index or else different lengths
                interaction_log_df.to_csv('data/pairwise_interaction_log{}.csv'.format(currentSimulation), index=False, header=True)
                
                
            elif INTERACTION_TYPE == 'group':
                interaction_log_df = pandas.DataFrame.from_dict(self.session.interaction_logs['group'], orient='index')
                interaction_log_df = interaction_log_df.transpose()
                interaction_log_df.to_csv('data/group_interaction_log{}.csv'.format(currentSimulation), index=False, header=True)
                
            
            sim_finish_time = time.time()
            
            print('Sim time taken: ', sim_finish_time - sim_start_time)
            
            print()
            
            currentSimulation = currentSimulation + 1
                
            

if __name__ == "__main__":

    start = time.time()
    #random.seed(51) # only reproducible for first sim if used here
    #np.random.seed(100) # no numpy randoms used
    print('Running')
    bbe = BBE(51) # seed passed to BBE so can be used in title of plots for race identification
    print('Running')
    bbe.runSession()
    end = time.time()
    print('Total Time taken: ', end - start)
    
    


# =============================================================================
# if __name__ == "__main__":
# 
#     for i in [22, 42, 51]:
#         #print('seed: ', i)
#         start = time.time()
#         random.seed(i)
#         #np.random.seed(i) # no numpy randoms used
#         print('Running')
#         bbe = BBE(i)
#         print('Running')
#         bbe.runSession()
#         end = time.time()
#         print('Time taken: ', end - start)
# =============================================================================
        

# =============================================================================
# 
# if __name__ == "__main__":
# 
#     for i in range(100):
#         #print('seed: ', i)
#         start = time.time()
#         random.seed(i)
#         #np.random.seed(i) # no numpy randoms used
#         print('Running')
#         bbe = BBE(i)
#         print('Running')
#         bbe.runSession()
#         end = time.time()
#         print('Time taken: ', end - start)
#         
# =============================================================================
        
