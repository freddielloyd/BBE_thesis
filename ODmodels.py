import random
from system_constants import *

import numpy as np
import pandas as pd

from network_structures import *
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

    def __init__(self, id, bettor1, bettor2, start_time, model, interaction_log):
        self.id = id
        self.bettor1 = bettor1
        self.bettor2 = bettor2
        self.start_time = start_time
        self.model = model
        self.conversation_length = random.uniform(2, 6)
        self.in_progress = 1
        self.bettor1.in_conversation = 1
        self.bettor2.in_conversation = 1
        
        self.interaction_log = interaction_log
        
        # bettors degree of connection will have changed when fuzzy step called so set here
        # to be able to append to interaction log correctly
        self.degree_of_connection = self.bettor2.degree_of_connection
        
        
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
        
        temp_pairwise_interaction_log = {'type': [], 
                                         'time': [], 
                                         'length': [],
                                         'bettor1': [], 
                                         'bettor1_id': [],
                                         'b1_local_op': [], 
                                         'bettor2': [],
                                         'bettor2_id': [],
                                         'deg_of_connection': [],
                                         'b2_local_op': [],
                                         'local_op_gap': [],
                                         'weight': [],
                                         'b1_new_local_op': [],
                                         'b2_new_local_op': []} 
        
        temp_pairwise_interaction_log['type'].append(self.model)                
        temp_pairwise_interaction_log['time'].append(self.start_time)
        temp_pairwise_interaction_log['length'].append(self.conversation_length)        
        temp_pairwise_interaction_log['bettor1'].append(str(self.bettor1).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
        temp_pairwise_interaction_log['bettor1_id'].append(self.bettor1.shuffled_id)    
        temp_pairwise_interaction_log['bettor2'].append(str(self.bettor2).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
        temp_pairwise_interaction_log['bettor2_id'].append(self.bettor2.shuffled_id)
        temp_pairwise_interaction_log['deg_of_connection'].append(self.degree_of_connection)  
        #print(self.bettor2.degree_of_connection)
    

        X_i = self.bettor1.local_opinion
        X_j = self.bettor2.local_opinion
        
        opinion_gap = abs(X_i - X_j)
        
        temp_pairwise_interaction_log['b1_local_op'].append(X_i)
        temp_pairwise_interaction_log['b2_local_op'].append(X_j)
        temp_pairwise_interaction_log['local_op_gap'].append(opinion_gap)

        # if difference in opinion is within deviation threshold
        if abs(X_i - X_j) <= delta:
            
            opinion_gap = abs(X_i - X_j)
                    
            fuzzy_bc = fuzzy_BC()
 
            w = fuzzy_bc.fuzzification(mfx, opinion_gap) # defuzzified agent interaction weight
            
            temp_pairwise_interaction_log['weight'].append(w)
            
            #print('defuzzified agent interaction weight: ', w)
            
            
            if self.bettor1.influenced_by_opinions == 1:
                i_update = (1 - w) * X_i + w * X_j # opinion is strength of weight times other agents opinion
                self.bettor1.set_opinion(i_update)
                temp_pairwise_interaction_log['b1_new_local_op'].append(i_update)
                
            if self.bettor2.influenced_by_opinions == 1:
                j_update = (1 - w) * X_j + w * X_i
                self.bettor2.set_opinion(j_update)
                temp_pairwise_interaction_log['b2_new_local_op'].append(j_update)
            elif self.bettor2.influenced_by_opinions == 0:
                temp_pairwise_interaction_log['b2_new_local_op'].append(X_j)
            
                
        elif abs(X_i - X_j) > delta:
            print('Opinion gap too far apart - no interaction occurs')

            temp_pairwise_interaction_log['weight'].append(0) # weight is essentially 0 if no update occurs
            temp_pairwise_interaction_log['b1_new_local_op'].append(X_i)
            temp_pairwise_interaction_log['b2_new_local_op'].append(X_j)
        
        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in temp_pairwise_interaction_log.items():
            self.interaction_log[key].append(value[0])

        
    
class GroupConversation:

    def __init__(self, id, bettor_initiator, group_bettors, start_time, model, interaction_log):
        
        self.id = id            
        
        self.bettor_initiator = bettor_initiator
        
        self.other_bettors = group_bettors

        self.start_time = start_time
        self.model = model
        self.conversation_length = random.uniform(2, 6)
        self.in_progress = 1
        
        for bettor in self.other_bettors:
            bettor.in_conversation = 1
            
        self.bettor_initiator.in_conversation = 1
        
        self.interaction_log = interaction_log
        
        # bettors degree of connection will have changed when fuzzy step called so set here
        # to be able to append to interaction log correctly
        self.degrees_of_connection = []
        for bettor in self.other_bettors:
            self.degrees_of_connection.append(bettor.degree_of_connection)
        
        
    
    def group_change_local_opinions(self):

        if self.model == 'fuzzy_BC':
            self.group_fuzzy_bounded_confidence_step(delta, FUZZY_MFX)  
            
        else:
            return print('Group OD model does not exist')

    # mfx is triangular or trapezoidal currently, interaction pairwise or group
    def group_fuzzy_bounded_confidence_step(self, delta, mfx):
               
        temp_group_interaction_log = {'type': [], 
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
                                           #'local_op_gap': [],
                                           'weights': [],
                                           'ops_x_weights': [],
                                           'b1_new_local_op': []}
       
        temp_group_interaction_log['type'].append(self.model)         
        temp_group_interaction_log['conv_id'].append(self.id)         
        temp_group_interaction_log['time'].append(self.start_time)
        temp_group_interaction_log['length'].append(self.conversation_length)        
        temp_group_interaction_log['bettor1'].append(str(self.bettor_initiator).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
        temp_group_interaction_log['bettor1_id'].append(self.bettor_initiator.shuffled_id)    
        
        reduced_names = []
        bettors_ids = []
        
        for bettor in self.other_bettors:
            reduced_names.append(str(bettor).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
            bettors_ids.append(bettor.shuffled_id)
        
        #print('group conv id: ', self.id, 'num other bettors: ', len(self.other_bettors))
        
        temp_group_interaction_log['num_bettors'].append(len(self.other_bettors))
        temp_group_interaction_log['bettors'].append(reduced_names)
        temp_group_interaction_log['bettors_ids'].append(bettors_ids)
        temp_group_interaction_log['degs_of_connection'].append(self.degrees_of_connection)
        
        #print('group conv id: ', self.id, 'bettor_initiator local op ', self.bettor_initiator.local_opinion)
        
        X_i = self.bettor_initiator.local_opinion
        #print('priveleged bettor local opinion: ', X_i)
        #print('group conv id: ', self.id, 'bettor_initiator local op ', self.bettor_initiator.local_opinion)
        #print('group conv id: ', self.id, 'bettor_initiator local op ', X_i)
        
        temp_group_interaction_log['b1_local_op'].append(X_i)
        
        
        self.group_local_opinions = [bettor.local_opinion for bettor in self.other_bettors]
        #print('group conv id: ', self.id, 'num other bettors: ', len(self.other_bettors))
        #print('group conv id: ', self.id, 'group local opinions: ', self.group_local_opinions)
        temp_group_interaction_log['bettors_local_ops'].append(self.group_local_opinions)

        self.dfz_weights = []
        
        #group_avg_opinion = np.mean(group_local_opinions)

        for bettor in self.other_bettors:
            
            X_j = bettor.local_opinion
            
            opinion_gap = abs(X_i - X_j)
            
            fuzzy_bc = fuzzy_BC()
 
            w = fuzzy_bc.fuzzification(mfx, opinion_gap) # defuzzified agent interaction weight
            
            self.dfz_weights.append(w)
            
        #for bettor in self.group_bettors:
            #if bettor.influenced_by_opinions == 1: # accounts for if there is more than one RP(d) bettor in conv
            
        self.X_i_updates = [self.group_local_opinions[i]*self.dfz_weights[i] for i in range(len(self.other_bettors))]
        
        
        #self.num_bettors = len(self.other_bettors)
        
        # self weight to be placed on new opinion calculation - opinion gap of zero
        self_weight = fuzzy_bc.fuzzification(mfx, 0) 
        self.dfz_weights.append(self_weight)
        temp_group_interaction_log['weights'].append(self.dfz_weights)
        
        self_update = X_i*self_weight
        self.X_i_updates.append(self_update)
        temp_group_interaction_log['ops_x_weights'].append(self.X_i_updates)
        
        self.new_xi_opinion = sum(self.X_i_updates)/sum(self.dfz_weights)
        
        
        #print('priveleged bettor local opinion: ', X_i)
        
        #print('group local opinions: ', group_local_opinions)
            
        
        #print('group conv id: ', self.id, 'dfz weights: ', self.dfz_weights)
        
        #print('Numerator: ', sum(X_i_updates)) 
        #print('Denominator: ', num_bettors)  
        
        # have been seting by number of bettors - THIS PRODUCES VERY LOW OUTPUTS FOR PRIV BETTORS
        #self.new_xi_opinion = sum(self.X_i_updates)/self.num_bettors
        #self.new_xi_opinion = sum(self.X_i_updates)/sum(self.dfz_weights)
        
        #print('group conv id: ', self.id, 'b1 new opinion: ', self.new_xi_opinion)

        temp_group_interaction_log['b1_new_local_op'].append(self.new_xi_opinion)
        
        # HAVE NEW OPINION TAKE INTO ACCOUNT OLD OPINION
        #new_opinion = 
        
        #print('new X_i opinion: ', new_xi_opinion)
        
        self.bettor_initiator.set_opinion(self.new_xi_opinion)
        
        
        #print('temp log: ', temp_group_interaction_log)
        #print()
        #print('actual log: ', self.interaction_log)
            
        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in temp_group_interaction_log.items():
            
            self.interaction_log[key].append(value)
            
       

class OpinionDynamicsPlatform:
    def __init__(self, bettors, model, network_structure, 
                 interaction_type, interaction_selection, interaction_logs):
        self.bettors = bettors
        
        self.model = model
        self.conversations = []
        self.number_of_conversations = 0
        
        self.network_structure = network_structure
        
        self.interaction_type = interaction_type
        if self.interaction_type == 'pairwise':
            self.interaction_log = interaction_logs['pairwise']
        elif self.interaction_type == 'group':
            self.interaction_log = interaction_logs['group']
            
        self.interaction_selection = interaction_selection

        self.all_influenced_by_opinions = [bettor for bettor in bettors if bettor.influenced_by_opinions == 1]
        self.all_opinionated = [bettor for bettor in bettors if bettor.opinionated == 1]
        

        self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                 bettor.in_conversation == 0]
        self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                      bettor.in_conversation == 0]
        
        # self.unavailable_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
        #                                            bettor.in_conversation == 1]
        # self.unavailable_opinionated = [bettor for bettor in self.all_opinionated if
        #                                 bettor.in_conversation == 1]
        
        
        # initialise network and output structure csv 
        if self.network_structure == 'watts_strogatz':
            
            watts_strogatz = WattsStrogatz(len(self.all_opinionated), num_neighbours, rewiring_prob)
            
            self.network = watts_strogatz.create_network()
            
            # create network structure data frame to output for use in Tableau
            vertexes = []
            bettor_types = []
            edges = []
            degrees = []
            
            for i in range(len(self.network.vertex())): 

                vertexes.append(np.sort(self.network.vertex())[i])
                bettor_types.append(str(self.all_opinionated[i]).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
                edges.append(self.network.degree(i))
                degrees.append(self.network.edge(i))
                
            data = {'id': [bettor.id for bettor in bettors],
                    'shuffled_id': vertexes,
                    'bettor type': bettor_types,
                    'Number of Neighbours': edges,
                    'Neighbours': degrees}
            
            df = pd.DataFrame(data)

            df.to_csv('/Users/freddielloyd/Documents/Uob Documents/DSP Thesis/data/network_structure.csv',
                      index = False)



    def initiate_conversations(self, time):
        
        if self.interaction_type == 'pairwise':
        
            if self.network_structure == 'fully_connected':
    
                for bettor in self.available_influenced_by_opinions:
        
                    bettor1 = bettor
                    bettor2 = bettor
                    
                    while bettor1 == bettor2:
                        if len(self.available_influenced_by_opinions) == 0 or len(self.available_opinionated) < 2:
                            return
                        else:
                            num_bettors_to_select = 1
                            bettor2 = random.sample(self.available_opinionated, num_bettors_to_select)[0]
                            
                    id = self.number_of_conversations
        
                    Conversation = LocalConversation(id, bettor1, bettor2, time, self.model, self.interaction_log)
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
                    
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                    
            elif self.network_structure == 'watts_strogatz':          
                
                #for bettor in self.available_influenced_by_opinions:
                    
                # changed to while loop to solve error of bettor2 being selected as another priv bettor
                # who was then iterated through in for loop, and therefore initiating interactions
                # while they were still in initial interaction, or vice versa in being selected for an interaction
                # having already initiated an interaction in the initial for loop
                while len(self.available_influenced_by_opinions) > 0:
                    
                    bettor = self.available_influenced_by_opinions[0]
        
                    bettor1 = bettor
                    bettor2 = bettor
                    
                    self.bettor1_id = self.all_opinionated.index(bettor1)
                    
                    # retrieve neighbour ids based on created network, implying id in list is its node id in network
                    self.bettor_neighbours_ids = self.network.edge(self.bettor1_id)

                    self.all_neighbours = []
                    
                    for i in self.bettor_neighbours_ids:
                        
                        self.all_neighbours.append(self.all_opinionated[i])
                    
                    
                    self.available_neighbours = [bettor for bettor in self.all_neighbours if
                                                  bettor.in_conversation == 0]
                    
            
                    while bettor1 == bettor2:
                        
                        if len(self.available_influenced_by_opinions) == 0 or len(self.available_neighbours) < 1:
                            return
                        else:
                            num_bettors_to_select = 1
                            bettor2 = self.select_network_interaction_bettors(self.interaction_type,
                                                                              self.interaction_selection,
                                                                              num_bettors_to_select)

  
                    id = self.number_of_conversations
                    
                    #self.bettor2_id = self.all_opinionated.index(bettor2)

                    print(bettor1)
                    print('bettor1 in conversation pre: ', bettor1.in_conversation)
                    print(bettor2)
                    print('bettor2 in conversation pre: ', bettor2.in_conversation)        

                    Conversation = LocalConversation(id, bettor1, bettor2, time, self.model, self.interaction_log)
                    
                    print(bettor1)
                    print('bettor1 in conversation post: ', bettor1.in_conversation)
                    print(bettor2)
                    print('bettor2 in conversation post: ', bettor2.in_conversation)
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                
                
                    
        elif self.interaction_type == 'group':
            
            if self.network_structure == 'fully_connected':
                
                for bettor in self.available_influenced_by_opinions:
                    
                    bettor1 = bettor

                    if len(self.available_influenced_by_opinions) == 0 or len(self.available_opinionated) < 2:
                        return
                    
                    elif len(self.available_opinionated) >= 2:
    
                        # number of other bettors to be in group conversation
                        num_bettors_to_select = random.randint(1, len(self.available_opinionated))
                        
                        conv_group = random.sample(self.available_opinionated, num_bettors_to_select) # rndomly select given amount of available neighbours
                    
                    # if bettor1 in the group, resample until not in group
                    while bettor1 in conv_group:
                        num_bettors_to_select = random.randint(1, len(self.available_opinionated))

                        conv_group = random.sample(self.available_opinionated, num_bettors_to_select)
                        
                            
                    id = self.number_of_conversations
                    
                    Conversation = GroupConversation(id, bettor1, conv_group, time, self.model, self.interaction_log)                
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                    
                
            elif self.network_structure == 'watts_strogatz':

                #print(self.available_influenced_by_opinions)
                
                for bettor in self.available_influenced_by_opinions:
                    
                    bettor1 = bettor
                    
                    bettor1_id = self.all_opinionated.index(bettor1)
                    
                    bettor_neighbours_ids = self.network.edge(bettor1_id)
                          
                    all_neighbours = []
                    
                    for i in bettor_neighbours_ids:
                        
                        all_neighbours.append(self.all_opinionated[i])
                        
                    
                    available_neighbours = [bettor for bettor in all_neighbours if
                                            bettor.in_conversation == 0]
                    
                    
                    if self.interaction_selection == 'direct_neighbours':
                        if len(self.available_influenced_by_opinions) == 0 or len(available_neighbours) < 1:
                            return
                        
                        elif len(available_neighbours) >= 1:
    
                            # number of other bettors to be in group conversation
                            num_bettors_to_select = random.randint(1, len(available_neighbours))
                            
                            conv_group = self.select_network_interaction_bettors(self.interaction_type,
                                                                         self.interaction_selection,
                                                                         num_bettors_to_select)
                            
                            #conv_group = random.sample(available_neighbours, num_bettors_in_conv)


                    elif self.interaction_selection == 'across_network':
                        if len(self.available_influenced_by_opinions) == 0 or len(self.available_opinionated) < 2:
                            return
                        else:
                            # number of other bettors to be in group conversation
                            num_bettors_to_select = random.randint(1, len(self.available_opinionated))

                            conv_group = self.select_network_interaction_bettors(self.interaction_type,
                                                                                 self.interaction_selection,
                                                                                 num_bettors_to_select)


                            
                    id = self.number_of_conversations
                    
                    Conversation = GroupConversation(id, bettor1, conv_group, time, self.model, self.interaction_log)
                    
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
                    
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                        


    def select_network_interaction_bettors(self, interaction_type, interaction_selection, num_bettors_to_select):
        
        self.interaction_type = interaction_type
        self.interaction_selection = interaction_selection
        
        num_bettors_to_select = num_bettors_to_select
        
        if self.interaction_selection == 'direct_neighbours':
            
            for bettor in self.all_opinionated:
                bettor.degree_of_connection = 1
            
            if self.interaction_type == 'pairwise':
                # randomly select one available neighbour
                bettor2 = random.sample(self.available_neighbours, num_bettors_to_select)[0]
                #bettor2.degree_of_connection = 1
                return bettor2
            
            if self.interaction_type == 'group':
                # randomly select given amount of available neighbours
                conv_group = random.sample(self.available_neighbours, num_bettors_to_select)[0]
                return conv_group
                
    
        elif self.interaction_selection == 'across_network':
        
            # iterate through all bettors and set degree of connection to bettor1 attribute
            for bettor in self.all_opinionated:
                
                bettor_id = self.all_opinionated.index(bettor)
                if bettor_id == self.bettor1_id:
                    bettor.degree_of_connection = 0
                    continue
                    
                bettor_neighbours_ids = self.network.edge(bettor_id)
                if self.bettor1_id in bettor_neighbours_ids:
                    bettor.degree_of_connection = 1
                    continue
                    
                #neighbours_neighbours = []
                for neighbour_id in bettor_neighbours_ids:
                    neighbours_neighbours_ids = self.network.edge(neighbour_id)
                    connection_found = False
                    if self.bettor1_id in neighbours_neighbours_ids:
                        bettor.degree_of_connection = 2
                        connection_found = True
                        break
                if connection_found == True:
                    continue
                
                for neighbour_neighbours_id in neighbours_neighbours_ids:
                    neighbours_neighbours_neighbours_ids = self.network.edge(neighbour_neighbours_id)
                    if self.bettor1_id in neighbours_neighbours_neighbours_ids:
                        bettor.degree_of_connection = 3
                        connection_found = True
                        break
                if connection_found == True:
                    continue
                
                
                else:
                    # if not 0,1,2 or 3 degree of connection then record 4+ as 4
                    bettor.degree_of_connection = 4
                
            probability_distribution = []
                
            for bettor in self.available_opinionated:
                if bettor.degree_of_connection == 0:
                    probability_distribution.append(0) # zero probability of selecting self
                elif bettor.degree_of_connection == 1:
                    probability_distribution.append(4/len(self.available_opinionated)) # highest prob
                elif bettor.degree_of_connection == 2:
                    probability_distribution.append(3/len(self.available_opinionated))
                elif bettor.degree_of_connection == 3:
                    probability_distribution.append(2/len(self.available_opinionated))
                elif bettor.degree_of_connection == 4:
                    probability_distribution.append(1/len(self.available_opinionated)) # lowest prob
                    
            #print(probability_distribution)
            #print(sum(probability_distribution))
                    
            # normalise probability distribution so sums to one
            norm_prob_dist = [prob/sum(probability_distribution) for prob in probability_distribution]
            #print(norm_prob_dist)
            print(sum(norm_prob_dist))
            
            if self.interaction_type == 'pairwise':
                # select one bettor randomly with given probability distribution
                bettor2 = np.random.choice(self.available_opinionated, num_bettors_to_select,
                                           p=norm_prob_dist)[0]
                return bettor2
            
            elif self.interaction_type == 'group':
                # select chosen number of bettors randomly with given probability distribution
                bettors = np.random.choice(self.available_opinionated, num_bettors_to_select,
                                           p=norm_prob_dist)[0]
                return bettors
            
 
                

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

    def update_opinions(self, time, markets):
        active_conversations = [c for c in self.conversations if c.in_progress == 1]

        # Update bettor local opinions (where conversation has reached an end)
        for c in active_conversations:

            if c.start_time + c.conversation_length <= time:

                
                if self.interaction_type == 'pairwise':
                    
                    c.change_local_opinions()
                    c.in_progress = 0
                
                    c.bettor1.in_conversation = 0
                    c.bettor2.in_conversation = 0
                    
                elif self.interaction_type == 'group':
                    
                    c.group_change_local_opinions()
                    c.in_progress = 0
                    
                    for bettor in c.other_bettors:
                        bettor.in_conversation = 0
                        
                    c.bettor_initiator.in_conversation = 0
                    

                self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                         bettor.in_conversation == 0]
                self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                              bettor.in_conversation == 0]

            else:
                continue

        # Update bettor global opinion, opinion weights, event opinion and finally calculate overall bettor opinion.
        for bettor in self.all_influenced_by_opinions:
            self.change_opinion(bettor, markets)



