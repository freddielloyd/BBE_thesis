import random
from system_constants import *
from network_structures import *

import numpy as np

from fuzzy_BC import *


def calculate_ema(odds, timesteps, smoothing=2):
    if len(odds) < timesteps:
        answer = 1 / NUM_OF_COMPETITORS
    else:
        ema = [sum(odds[:timesteps]) / timesteps]
        for odd in odds[timesteps:]:
            ema.append((odd * (smoothing / (1 + timesteps))) + ema[-1] * (1 - (smoothing / (1 + timesteps))))
        answer = 1 / ema[-1]
    return answer


class LocalConversation:

    def __init__(self, id, bettor1, bettor2, start_time, model):
        self.id = id
        self.bettor1 = bettor1
        self.bettor2 = bettor2
        self.start_time = start_time
        self.model = model
        self.conversation_length = random.uniform(2, 6)
        self.in_progress = 1
        self.bettor1.in_conversation = 1
        self.bettor2.in_conversation = 1

    def change_local_opinions(self):
        if self.model == 'BC':
            self.bounded_confidence_step(mu, delta)
        elif self.model == 'RA':
            self.relative_agreement_step(mu)
        elif self.model == 'RD':
            self.relative_disagreement_step(mu, lmda)
            
        elif self.model == 'fuzzy_BC':
            self.fuzzy_bounded_confidence_step(delta, FUZZY_MFX)  
            
        else:
            return print('OD model does not exist')

    # # Opinion dynamics models

    # bounded confidence model
    # (w, delta) are confidence factor and deviation threshold  respectively
    def bounded_confidence_step(self, w, delta):

        X_i = self.bettor1.local_opinion
        X_j = self.bettor2.local_opinion

        # if difference in opinion is within deviation threshold
        if abs(X_i - X_j) <= delta:
            if self.bettor1.influenced_by_opinions == 1:
                i_update = w * X_i + (1 - w) * X_j
                self.bettor1.set_opinion(i_update)
            if self.bettor2.influenced_by_opinions == 1:
                j_update = w * X_j + (1 - w) * X_i
                self.bettor2.set_opinion(j_update)
            


    def relative_agreement_step(self, weight):

        X_i = self.bettor1.local_opinion
        u_i = self.bettor1.uncertainty

        X_j = self.bettor2.local_opinion
        u_j = self.bettor2.uncertainty

        h_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
        h_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))

        if (h_ji > u_j):
            if self.bettor1.influenced_by_opinions == 1:
                RA_ji = (h_ji / u_j) - 1
                self.bettor1.set_opinion(X_i + (weight * RA_ji * (X_j - X_i)))
                self.bettor1.set_uncertainty(u_i + (weight * RA_ji * (u_j - u_i)))
        if (h_ij > u_i):
            if self.bettor2.influenced_by_opinions == 1:
                RA_ij = (h_ij / u_i) - 1
                self.bettor2.set_opinion(X_j + (weight * RA_ij * (X_i - X_j)))
                self.bettor2.set_uncertainty(u_j + (weight * RA_ij * (u_i - u_j)))

    def relative_disagreement_step(self, weight, prob):

        X_i = self.bettor1.local_opinion
        u_i = self.bettor1.uncertainty

        X_j = self.bettor2.local_opinion
        u_j = self.bettor2.uncertainty

        # Calculate overlap
        # g_ij = max((X_i - u_i), (X_j - u_j)) - min((X_i + u_i), (X_j + u_j))
        g_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
        # g_ji = max((X_j - u_j), (X_i - u_i)) - min((X_j + u_j), (X_i + u_i))
        g_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))

        # update with prob λ
        if random.random() <= prob:
            if (g_ji > u_j):
                if self.bettor1.influenced_by_opinions == 1:
                    RD_ji = (g_ji / u_j) - 1
                    self.bettor1.set_opinion(X_i - (weight * RD_ji * (X_j - X_i)))
                    self.bettor1.set_uncertainty(u_i + (weight * RD_ji * (u_j - u_i)))
            if (g_ij > u_i):
                if self.bettor2.influenced_by_opinions == 1:
                    RD_ij = (g_ij / u_i) - 1
                    self.bettor2.set_opinion(X_j - (weight * RD_ij * (X_i - X_j)))
                    self.bettor2.set_uncertainty(u_j + (weight * RD_ij * (u_i - u_j)))
                                  
                    
    # mfx is triangular or trapezoidal currently
    def fuzzy_bounded_confidence_step(self, delta, mfx):
    
    #if interactions == 'pairwise':

        X_i = self.bettor1.local_opinion
        X_j = self.bettor2.local_opinion

        # if difference in opinion is within deviation threshold
        if abs(X_i - X_j) <= delta:
            
            opinion_gap = abs(X_i - X_j)
            fuzzy_bc = fuzzy_BC()
 
            w = fuzzy_bc.fuzzification(mfx, opinion_gap) # defuzzified agent interaction weight
            
            print('defuzzified agent interaction weight: ', w)
            
            
            if self.bettor1.influenced_by_opinions == 1:
                i_update = (1 - w) * X_i + w * X_j # opinion is strength of weight times other agents opinion
                self.bettor1.set_opinion(i_update)
            if self.bettor2.influenced_by_opinions == 1:
                j_update = (1 - w) * X_j + w * X_i
                self.bettor2.set_opinion(j_update)
                
        elif abs(X_i - X_j) > delta:
            print('Opinion gap too far apart - no interaction occurs')
            
    #elif interactions == 'group':
        
        
    #    pass
        
    
class GroupConversation:

    def __init__(self, id, bettors, start_time, model):
        
        self.id = id    
        
        #bettors = conv_group
        
        bettor_initiator = bettors[-1] # last one of group will always be the initiator
        
        self.bettors = bettors
        
# =============================================================================
#         bettor_dict = {}
#         for i, item in enumerate(bettors):
#             bettor_dict['bettor' + str(i+1)] = item
#             
#         bettor_dict.items()
#         bettor_dict.values()
# =============================================================================

        self.start_time = start_time
        self.model = model
        self.conversation_length = random.uniform(2, 6)
        self.in_progress = 1
        
        for bettor in bettors:
            bettor.in_conversation = 1
            
        print(bettor_initiator.in_conversation) # should be 1
            
            
    
    def group_change_local_opinions(self):

            
        if self.model == 'fuzzy_BC':
            self.group_fuzzy_bounded_confidence_step(delta, FUZZY_MFX)  
            
        else:
            return print('Group OD model does not exist')
            
                    
    # mfx is triangular or trapezoidal currently, interaction pairwise or group
    def group_fuzzy_bounded_confidence_step(self, delta, mfx):
        
        group_local_opinions = [bettor.local_opinion for bettor in self.bettors]
        
        # local opinion of opinion influenced bettor as is always the last one
        X_i = group_local_opinions[-1]

        
        opinions_within_threshold = []
        for j in range(len(self.bettors)-1):
            X_j = group_local_opinions[j]
            
            if abs(X_i - X_j) <= delta:
                opinions_within_threshold.append(X_j) # NOT EXACTLY RIGHT NEEDS ALTERING AS CANT THEN ACCESS WHICH BETTORS
            

        # if difference in opinion is within deviation threshold
        if abs(X_i - X_j) <= delta:
            
            opinion_gap = abs(X_i - X_j)
            fuzzy_bc = fuzzy_BC()
 
            w = fuzzy_bc.fuzzification(mfx, opinion_gap) # defuzzified agent interaction weight
            
            print('defuzzified agent interaction weight: ', w)
            
            
            if self.bettor1.influenced_by_opinions == 1:
                i_update = (1 - w) * X_i + w * X_j # opinion is strength of weight times other agents opinion
                self.bettor1.set_opinion(i_update)
            if self.bettor2.influenced_by_opinions == 1:
                j_update = (1 - w) * X_j + w * X_i
                self.bettor2.set_opinion(j_update)
                
        elif abs(X_i - X_j) > delta:
            print('Opinion gap too far apart - no interaction occurs')

    
    
    
        



class OpinionDynamicsPlatform:
    def __init__(self, bettors, model, network_structure, interactions):
        self.bettors = bettors
        self.model = model
        self.conversations = []
        self.number_of_conversations = 0
        
        self.network_structure = network_structure
        
        self.interactions = interactions

        self.all_influenced_by_opinions = [bettor for bettor in bettors if bettor.influenced_by_opinions == 1]
        self.all_opinionated = [bettor for bettor in bettors if bettor.opinionated == 1]

        self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                 bettor.in_conversation == 0]
        self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                      bettor.in_conversation == 0]
        
        #
        # self.unavailable_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
        #                                            bettor.in_conversation == 1]
        # self.unavailable_opinionated = [bettor for bettor in self.all_opinionated if
        #                                 bettor.in_conversation == 1]
        
        if self.network_structure == 'watts_strogatz':
            
            watts_strogatz = WattsStrogatz(len(self.all_opinionated), num_neighbours, rewiring_prob)
            
            self.network = watts_strogatz.create_network()
            
            
            #network.vertex()
            #network.degree(5)
            
            for i in range(len(self.network.vertex())): 

                print('node:', np.sort(self.network.vertex())[i], ', degree:', self.network.degree(i), ', edges: ', self.network.edge(i))



    def initiate_conversations(self, time):
        
        if self.network_structure == 'fully_connected':

            for bettor in self.available_influenced_by_opinions:
    
                bettor1 = bettor
                bettor2 = bettor
    
                while bettor1 == bettor2:
                    if len(self.available_influenced_by_opinions) == 0 or len(self.available_opinionated) < 2:
                        return
                    else:
                        bettor2 = random.sample(self.available_opinionated, 1)[0]
    
                id = self.number_of_conversations
    
                Conversation = LocalConversation(id, bettor1, bettor2, time, self.model)
    
                self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                         bettor.in_conversation == 0]
                self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                              bettor.in_conversation == 0]
                
    
                self.conversations.append(Conversation)
                self.number_of_conversations = self.number_of_conversations + 1
            
            
        if self.network_structure == 'watts_strogatz':
            
            if self.interactions == 'pairwise':
            
                
                for bettor in self.available_influenced_by_opinions:
        
                    bettor1 = bettor
                    bettor2 = bettor
                    
                    self.bettor1_id = self.all_opinionated.index(bettor1)
                    
                    #print(self.bettor1_id)
                    # retrieve neighbour ids based on created network, implying id in list is its node id in network
                    self.bettor_neighbours_ids = self.network.edge(self.bettor1_id)
                    #print(self.bettor_neighbours_ids)
                    
                    
                    
                    self.all_neighbours = []
                    
                    for i in self.bettor_neighbours_ids:
                        
                        self.all_neighbours.append(self.all_opinionated[i])
                    
                    #print(self.available_neighbours)
                    
                    self.available_neighbours = [bettor for bettor in self.all_neighbours if
                                                  bettor.in_conversation == 0]
                    
                    
        
                    while bettor1 == bettor2:
                        if len(self.available_influenced_by_opinions) == 0 or len(self.available_neighbours) < 2:
                            return
                        else:
                            bettor2 = random.sample(self.available_neighbours, 1)[0] # randomly select one available neighbour
    
    
                    id = self.number_of_conversations
                    
                    self.bettor2_id = self.all_opinionated.index(bettor2)
                    
                    
                    
                    print('bettor1: ', bettor1)
                    print('bettor1 id: ', self.bettor1_id)
                    print('bettor1 neighbours: ', self.bettor_neighbours_ids)
                    
                    print('bettor2: ', bettor2)
                    print('bettor2 id: ', self.bettor2_id)
                    #print('bettor2 neighbours: ', self.bettor_neighbours_ids)
                    
                    print()
        
                    Conversation = LocalConversation(id, bettor1, bettor2, time, self.model)
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    #self.available_neighbours = [bettor for bettor in self.all_opinionated if
                    #                              bettor.in_conversation == 0]
                    
                    #self.available_opinionated = [bettor for bettor in self.all_opinionated if
                    #                              bettor.in_conversation == 0]
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
    

            elif self.interactions == 'group':
                
                for bettor in self.available_influenced_by_opinions:
                    
                    bettor1 = bettor
                    
                    bettor1_id = self.all_opinionated.index(bettor1)
                    
                    bettor_neighbours_ids = self.network.edge(bettor1_id)
                    
                          
                    self.all_neighbours = []
                    
                    for i in bettor_neighbours_ids:
                        
                        self.all_neighbours.append(self.all_opinionated[i])
                    
                    #print(self.available_neighbours)
                    
                    available_neighbours = [bettor for bettor in self.all_neighbours if
                                            bettor.in_conversation == 0]
                    
                    
                         
                    while len(available_neighbours) > 0:
                        if len(self.available_influenced_by_opinions) == 0 or len(available_neighbours) < 2:
                            return
                        else:
                            # number of bettors to be in group conversation
                            num_bettors_in_conv = random.randint(1, len(available_neighbours))
                            
                            conv_group = random.sample(available_neighbours, num_bettors_in_conv) # rndomly select given amount of available neighbours
                        
                    id = self.number_of_conversations
                    
                    
                    #group_bettor_ids = []
                    #for bettor in conv_group:
                    #    bettor_id = self.all_opinionated.index(bettor)
                    #                    
                    #    group_bettor_ids.append(bettor_id)
                    #conv_bettor_ids = bettor1_id.extend(group_bettor_ids)
                    
                    conv_group.append(bettor1) # retrieve entire selection of bettors to be in conversation
                    
                    Conversation = GroupConversation(id, conv_group, time, model)
                    
                
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    #self.available_opinionated = [bettor for bettor in self.all_opinionated if
                    #                              bettor.in_conversation == 0]
                    
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                        

                    
                
                

    def settle_opinions(self, winningCompetitor):

        for bettor in self.all_influenced_by_opinions:

            bettor.a3 = 1
            bettor.a2 = 0
            bettor.a1 = 0

            if OPINION_COMPETITOR == winningCompetitor:
                bettor.event_opinion = 1
            else:
                bettor.event_opinion = 0

            bettor.opinion = bettor.a1 * bettor.local_opinion + bettor.a2 * bettor.global_opinion + bettor.a3 * bettor.event_opinion

    def change_opinion(self, bettor, markets):

        if len(bettor.currentRaceState) > 0:

            # Update opinion weights a1, a2, a3

            bettor.a3 = max(bettor.currentRaceState.values()) / bettor.lengthOfRace
            a2 = (1 - bettor.a3) * bettor.start_a2
            bettor.a2 = a2
            bettor.a1 = 1 - bettor.a3 - bettor.a2

            if round(bettor.a1 + bettor.a2 + bettor.a3, 0) != 1:
                print('Warning: the starting weights of opinions are incorrect. '
                      '(bettor.a1, bettor.a2 and bettor.a3 should add up to 1): ',
                      round(bettor.a1 + bettor.a2 + bettor.a3, 0))
                print('\n bettor.a1: ', bettor.a1)
                print('\n bettor.a2: ', bettor.a2)
                print('\n bettor.a3: ', bettor.a3)


            # Update global opinion

            odds = [x for i, x in enumerate(bettor.competitor_odds['odds']) if
                    bettor.competitor_odds['competitor'][i] == OPINION_COMPETITOR]

            bettor.global_opinion = calculate_ema(odds, 80)

            # Update event opinion
            if bettor.bettingPeriod:
                total = 0
                for c in bettor.currentRaceState.values():
                    total = total + (bettor.lengthOfRace / (bettor.lengthOfRace - c)) ** 2

                bettor.event_opinion = (bettor.lengthOfRace / (
                    bettor.lengthOfRace - bettor.currentRaceState[OPINION_COMPETITOR])) ** 2 / total

        # Update overall bettor opinion
        bettor.opinion = bettor.a1 * bettor.local_opinion + \
                         bettor.a2 * bettor.global_opinion + bettor.a3 * bettor.event_opinion

    def update_opinions(self, time, markets, INTERACTIONS):
        active_conversations = [c for c in self.conversations if c.in_progress == 1]

        # Update bettor local opinions (where conversation has reached an end)
        for c in active_conversations:

            if c.start_time + c.conversation_length <= time:

                
                if INTERACTIONS == 'pairwise':
                    
                    c.change_local_opinions()
                    c.in_progress = 0
                
                    c.bettor1.in_conversation = 0
                    c.bettor2.in_conversation = 0
                    
                elif INTERACTIONS == 'group':
                    
                    c.group_change_local_opinions()
                    c.in_progress = 0
                    
                    for bettor in self.bettors:
                        bettor.in_conversation = 0
                    

                self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                         bettor.in_conversation == 0]
                self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                              bettor.in_conversation == 0]

            else:
                continue

        # Update bettor global opinion, opinion weights, event opinion and finally calculate overall bettor opinion.
        for bettor in self.all_influenced_by_opinions:
            self.change_opinion(bettor, markets)
