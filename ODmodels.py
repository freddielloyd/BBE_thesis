import random
from system_constants import *

import numpy as np
import pandas as pd

from network_structures import *
from fuzzy_logic import *


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

    def __init__(self, id, bettor1, bettor2, start_time, model, interaction_log, muddle_opinions):
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
        
        self.muddle_opinions = muddle_opinions
        
        # bettors degree of connection will have changed when change_local_opinions called 
        # so set here to be able to append to interaction log correctly
        self.degree_of_connection = self.bettor2.degree_of_connection
        
        
        self.temp_pairwise_interaction_log = {'type': [], 
                                             'time': [], 
                                             'length': [],
                                             'bettor1': [], 
                                             'bettor1_id': [],
                                             'b1_local_op': [], 
                                             'b1_uncertainty': [],
                                             'bettor2': [],
                                             'bettor2_id': [],
                                             'deg_of_connection': [],
                                             'b2_local_op': [],
                                             'b2_uncertainty': [],
                                             'b2_expressed_op': [],
                                             'local_op_gap': [],
                                             'weight': [],
                                             'b1_new_local_op': [],
                                             'b1_op_change': [],
                                             'b2_new_local_op': []} 
        
        
    def change_local_opinions(self):
                    
        self.temp_pairwise_interaction_log['type'].append(self.model)                
        self.temp_pairwise_interaction_log['time'].append(round(self.start_time, 2))
        self.temp_pairwise_interaction_log['length'].append(round(self.conversation_length, 2))        
        self.temp_pairwise_interaction_log['bettor1'].append(str(self.bettor1).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
        self.temp_pairwise_interaction_log['bettor1_id'].append(self.bettor1.shuffled_id)    
        self.temp_pairwise_interaction_log['bettor2'].append(str(self.bettor2).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
        self.temp_pairwise_interaction_log['bettor2_id'].append(self.bettor2.shuffled_id)
        self.temp_pairwise_interaction_log['deg_of_connection'].append(self.degree_of_connection) 
        
        self.temp_pairwise_interaction_log['b1_local_op'].append(round(self.bettor1.local_opinion, 2))
        self.temp_pairwise_interaction_log['b2_local_op'].append(round(self.bettor2.local_opinion, 2))
        
        self.temp_pairwise_interaction_log['b1_uncertainty'].append(round(self.bettor1.uncertainty, 2))
        self.temp_pairwise_interaction_log['b2_uncertainty'].append(round(self.bettor2.uncertainty, 2))
        
        if self.model == 'BC':
            self.bounded_confidence_step(mu, delta)
        elif self.model == 'RA':
            self.relative_agreement_step(mu)
        elif self.model == 'RD':
            self.relative_disagreement_step(mu, lmda)
            
        elif self.model == 'fuzzy_BC':
            self.fuzzy_bounded_confidence_step(FUZZY_MFX)  
            
        else:
            return print('OD model does not exist')

    # # Opinion dynamics models

    # bounded confidence model
    # (w, delta) are confidence factor and deviation threshold  respectively
    def bounded_confidence_step(self, w, delta):

        X_i = self.bettor1.local_opinion
        X_j = self.bettor2.local_opinion
        
        if self.muddle_opinions == 'yes':
            X_j = self.ambiguous_opinion(X_j, amount = 'medium')
            self.temp_pairwise_interaction_log['b2_expressed_op'].append(round(X_j, 2))
        elif self.muddle_opinions == 'no':
            self.temp_pairwise_interaction_log['b2_expressed_op'].append('N/A')
            
        opinion_gap = abs(X_i - X_j)
        self.temp_pairwise_interaction_log['local_op_gap'].append(round(opinion_gap, 2))
        #self.temp_pairwise_interaction_log['weight'].append(w)

        # if difference in opinion is within deviation threshold
        if opinion_gap <= delta:
            self.temp_pairwise_interaction_log['weight'].append(w)
            if self.bettor1.influenced_by_opinions == 1:
                i_update = w * X_i + (1 - w) * X_j
                self.bettor1.set_opinion(i_update)
                self.temp_pairwise_interaction_log['b1_new_local_op'].append(round(i_update, 2))
                self.temp_pairwise_interaction_log['b1_op_change'].append(round(i_update - X_i, 2))
                
            if self.bettor2.influenced_by_opinions == 1:
                j_update = w * X_j + (1 - w) * X_i
                self.bettor2.set_opinion(j_update)
                self.temp_pairwise_interaction_log['b2_new_local_op'].append(round(j_update, 2))
            elif self.bettor2.influenced_by_opinions == 0:
                self.temp_pairwise_interaction_log['b2_new_local_op'].append('N/A')
                
        
        elif opinion_gap > delta:
        #    print('Opinion gap too far apart - no interaction occurs')
            self.temp_pairwise_interaction_log['weight'].append(0) # weight is essentially 0 if no update occurs
            self.temp_pairwise_interaction_log['b1_new_local_op'].append('N/A')
            self.temp_pairwise_interaction_log['b1_op_change'].append(0)
            self.temp_pairwise_interaction_log['b2_new_local_op'].append('N/A')
        
        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in self.temp_pairwise_interaction_log.items():
            self.interaction_log[key].append(value[0])
            


    def relative_agreement_step(self, weight):
        
        # currently all agents initialised with same uncertainty, so u_i = u_j so no update of 
        # uncertainty can ever happen


        X_i = self.bettor1.local_opinion
        u_i = self.bettor1.uncertainty

        X_j = self.bettor2.local_opinion
        u_j = self.bettor2.uncertainty
        
        if self.muddle_opinions == 'yes':
            X_j = self.ambiguous_opinion(X_j, amount = 'medium')
            self.temp_pairwise_interaction_log['b2_expressed_op'].append(round(X_j, 2))
        elif self.muddle_opinions == 'no':
            self.temp_pairwise_interaction_log['b2_expressed_op'].append('N/A')
            
        opinion_gap = abs(X_i - X_j)  
        self.temp_pairwise_interaction_log['local_op_gap'].append(round(opinion_gap, 2)) 
        

        # symmetrical so these are the same?
        h_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
        h_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))
        
        #print('xi local op: ', X_i)
        #print('xj local op: ', X_j)
        
        #print('xi uncertainty: ', u_i)
        #print('xj uncertainty: ', u_j)
        
        #print('hij: ', h_ij)
        #print('hji: ', h_ji)

        if (h_ji > u_j):
            self.temp_pairwise_interaction_log['weight'].append(weight)
            if self.bettor1.influenced_by_opinions == 1:
                RA_ji = (h_ji / u_j) - 1
                print('RA_ji: ', RA_ji)
                i_update = X_i + (weight * RA_ji * (X_j - X_i))
                self.bettor1.set_opinion(i_update)
                uncertainity_update = u_i + (weight * RA_ji * (u_j - u_i))
                self.bettor1.set_uncertainty(uncertainity_update)
                
                self.temp_pairwise_interaction_log['b1_new_local_op'].append(round(i_update, 2))
                self.temp_pairwise_interaction_log['b1_op_change'].append(round(i_update - X_i, 2))
        elif (h_ji <= u_j):
                self.temp_pairwise_interaction_log['weight'].append(0)
                self.temp_pairwise_interaction_log['b1_new_local_op'].append('N/A')
                self.temp_pairwise_interaction_log['b1_op_change'].append(0)
            
                
        if (h_ij > u_i):
            if self.bettor2.influenced_by_opinions == 1:
                RA_ij = (h_ij / u_i) - 1
                j_update = X_j + (weight * RA_ij * (X_i - X_j))
                self.bettor2.set_opinion(j_update)
                self.bettor2.set_uncertainty(u_j + (weight * RA_ij * (u_i - u_j)))
                
                self.temp_pairwise_interaction_log['b2_new_local_op'].append(round(j_update, 2))
            elif self.bettor2.influenced_by_opinions == 0:
                self.temp_pairwise_interaction_log['b2_new_local_op'].append('N/A')
        elif (h_ij <= u_i):
                self.temp_pairwise_interaction_log['b2_new_local_op'].append('N/A')
            
        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in self.temp_pairwise_interaction_log.items():
            self.interaction_log[key].append(value[0])

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
                                  
                    
    # mfx is triangular currently
    def fuzzy_bounded_confidence_step(self, mfx):

        X_i = self.bettor1.local_opinion
        X_j = self.bettor2.local_opinion
        
        if self.muddle_opinions == 'yes':
            X_j = self.ambiguous_opinion(X_j, amount = 'high')
            self.temp_pairwise_interaction_log['b2_expressed_op'].append(round(X_j, 2))
        elif self.muddle_opinions == 'no':
            self.temp_pairwise_interaction_log['b2_expressed_op'].append('N/A')
        
        opinion_gap = abs(X_i - X_j)
        self.temp_pairwise_interaction_log['local_op_gap'].append(round(opinion_gap, 2))
                
        fuzzy_system = fuzzy_logic()
        # weight segmentations are a,b,c,d for researchers fuzzy rules, a1 for designed rules
        fuzzy_set = fuzzy_system.fuzzification(mfx, opinion_gap, weight_segmentation = 'a1')
        w = fuzzy_system.defuzzification(fuzzy_set, method = 'centroid')
        self.temp_pairwise_interaction_log['weight'].append(round(w, 2))
        
        if self.bettor1.influenced_by_opinions == 1:
            i_update = (1 - w) * X_i + w * X_j # opinion is strength of weight times other agents opinion
            self.bettor1.set_opinion(i_update)
            self.temp_pairwise_interaction_log['b1_new_local_op'].append(round(i_update, 2))
            self.temp_pairwise_interaction_log['b1_op_change'].append(round(i_update - X_i, 2))
            
        if self.bettor2.influenced_by_opinions == 1:
            j_update = w * X_j + (1 - w) * X_i
            self.bettor2.set_opinion(j_update)
            self.temp_pairwise_interaction_log['b2_new_local_op'].append(round(j_update, 2))
        elif self.bettor2.influenced_by_opinions == 0:
            self.temp_pairwise_interaction_log['b2_new_local_op'].append(round(X_j, 2))
            
        
        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in self.temp_pairwise_interaction_log.items():
            self.interaction_log[key].append(value[0])
 
        
    def ambiguous_opinion(self, exact_opinion, amount):
        
        if amount == 'medium':
            # strongest opinions either poisitive or negative have least ambiguity 
            if 0 <= exact_opinion <= 0.2:
                expressed_opinion = round(max(random.uniform(exact_opinion-0.05, exact_opinion+0.05), 0), 2)
            elif 0.8 <= exact_opinion <= 1:
                expressed_opinion = round(min(random.uniform(exact_opinion-0.05, exact_opinion+0.05), 1), 2)
            # reasonably strong opinions have slightly more ambiguity
            elif (0.2 < exact_opinion <= 0.4) or (0.6 <= exact_opinion < 0.8):
                expressed_opinion = round(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 2)  
            # neutral opinions have greatest ambiguity
            elif 0.4 < exact_opinion < 0.6:
                expressed_opinion = round(random.uniform(exact_opinion-0.15, exact_opinion+0.15), 2)
                
            return expressed_opinion
        
        elif amount == 'low':
            # strongest opinions either poisitive or negative have least ambiguity 
            if 0 <= exact_opinion <= 0.2:
                expressed_opinion = round(max(random.uniform(exact_opinion-0.02, exact_opinion+0.02), 0), 2)
            elif 0.8 <= exact_opinion <= 1:
                expressed_opinion = round(min(random.uniform(exact_opinion-0.02, exact_opinion+0.02), 1), 2)
            # reasonably strong opinions have slightly more ambiguity
            elif (0.2 < exact_opinion <= 0.4) or (0.6 <= exact_opinion < 0.8):
                expressed_opinion = round(random.uniform(exact_opinion-0.05, exact_opinion+0.05), 2)
            # neutral opinions have greatest ambiguity
            elif 0.4 < exact_opinion < 0.6:
                expressed_opinion = round(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 2)
                
            return expressed_opinion
        
        elif amount == 'high':
            # strongest opinions either poisitive or negative have least ambiguity 
            if 0 <= exact_opinion <= 0.2:
                expressed_opinion = round(max(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 0), 2)
            elif 0.8 <= exact_opinion <= 1:
                expressed_opinion = round(min(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 1), 2)
            # reasonably strong opinions have slightly more ambiguity
            elif (0.2 < exact_opinion <= 0.4) or (0.6 <= exact_opinion < 0.8):
                expressed_opinion = round(random.uniform(exact_opinion-0.18, exact_opinion+0.18), 2)  
            # neutral opinions have greatest ambiguity
            elif 0.4 < exact_opinion < 0.6:
                expressed_opinion = round(random.uniform(exact_opinion-0.25, exact_opinion+0.25), 2)
                
            return expressed_opinion

        
    
class GroupConversation:

    def __init__(self, id, bettor_initiator, group_bettors, start_time, model, interaction_log, muddle_opinions):
        
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
        
        self.muddle_opinions = muddle_opinions
        
        # bettors degree of connection will have changed when fuzzy step called so set here
        # to be able to append to interaction log correctly
        self.degrees_of_connection = []
        for bettor in self.other_bettors:
            self.degrees_of_connection.append(bettor.degree_of_connection)
            
        self.temp_group_interaction_log = {'type': [], 
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
        
        
    
    def group_change_local_opinions(self):
        
               
        self.temp_group_interaction_log['type'].append(self.model)         
        self.temp_group_interaction_log['conv_id'].append(self.id)         
        self.temp_group_interaction_log['time'].append(round(self.start_time, 2))
        self.temp_group_interaction_log['length'].append(round(self.conversation_length, 2))        
        self.temp_group_interaction_log['bettor1'].append(str(self.bettor_initiator).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
        self.temp_group_interaction_log['bettor1_id'].append(self.bettor_initiator.shuffled_id)    
        
        reduced_names = []
        bettors_ids = []
        
        for bettor in self.other_bettors:
            reduced_names.append(str(bettor).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
            bettors_ids.append(bettor.shuffled_id)
        
        #print('group conv id: ', self.id, 'num other bettors: ', len(self.other_bettors))
        
        self.temp_group_interaction_log['num_bettors'].append(len(self.other_bettors))
        self.temp_group_interaction_log['bettors'].append(reduced_names)
        self.temp_group_interaction_log['bettors_ids'].append(bettors_ids)
        self.temp_group_interaction_log['degs_of_connection'].append(self.degrees_of_connection)

        if self.model == 'BC':
            self.group_bounded_confidence_step(mu, delta) 
        elif self.model == 'fuzzy_BC':
            self.group_fuzzy_bounded_confidence_step(FUZZY_MFX)  
            
        else:
            return print('Group OD model does not exist')
        
    def group_bounded_confidence_step(self, mu, delta):
        
        X_i = self.bettor_initiator.local_opinion
        self.temp_group_interaction_log['b1_local_op'].append(round(X_i, 2))
        
        
        group_local_opinions = [round(bettor.local_opinion, 2) for bettor in self.other_bettors]
        self.temp_group_interaction_log['bettors_local_ops'].append(group_local_opinions)
        
        if self.muddle_opinions == 'yes':
            group_local_opinions = [round(self.ambiguous_opinion(bettor.local_opinion), 2) for bettor in self.other_bettors]
            self.temp_group_interaction_log['bettors_expressed_ops'].append(group_local_opinions)    
        else:
            self.temp_group_interaction_log['bettors_expressed_ops'].append(group_local_opinions)

        weights = []

        for local_op in group_local_opinions:
            opinion_gap = abs(X_i - local_op)
    
            # if difference in opinion is within deviation threshold
            if opinion_gap <= delta:
                weights.append(1)   
            elif opinion_gap > delta:
                weights.append(0)
                
        ops_x_weights = [group_local_opinions[i]*weights[i] for i in range(len(self.other_bettors))]
                
        self_weight = 1
        weights.append(self_weight)
        self.temp_group_interaction_log['weights'].append(weights)
        
        self_update = X_i*self_weight
        ops_x_weights.append(self_update)
        self.temp_group_interaction_log['ops_x_weights'].append(ops_x_weights)
        
        num_non_zero_weights = weights.count(1)
        
        new_xi_opinion = sum(ops_x_weights)/num_non_zero_weights
        
        self.temp_group_interaction_log['b1_new_local_op'].append(round(new_xi_opinion, 2))
        
        self.temp_group_interaction_log['b1_op_change'].append(round(new_xi_opinion - X_i, 2))
        
        
        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in self.temp_group_interaction_log.items():
            self.interaction_log[key].append(value[0])
            

        

    # mfx is triangular currently
    def group_fuzzy_bounded_confidence_step(self, mfx):
               
        X_i = self.bettor_initiator.local_opinion
        self.temp_group_interaction_log['b1_local_op'].append(round(X_i, 2))
        
        
        group_local_opinions = [round(bettor.local_opinion, 2) for bettor in self.other_bettors]
        self.temp_group_interaction_log['bettors_local_ops'].append(group_local_opinions)
        
        if self.muddle_opinions == 'yes':
            group_local_opinions = [round(self.ambiguous_opinion(bettor.local_opinion), 2) for bettor in self.other_bettors]
            self.temp_group_interaction_log['bettors_expressed_ops'].append(group_local_opinions)    
        else:
            self.temp_group_interaction_log['bettors_expressed_ops'].append(group_local_opinions)
        

        dfz_weights = []

        for bettor in self.other_bettors:
            X_j = bettor.local_opinion
            opinion_gap = abs(X_i - X_j)
            fuzzy_system = fuzzy_logic()
            fuzzy_set = fuzzy_system.fuzzification(mfx, opinion_gap, weight_segmentation = 'b')
            w = fuzzy_system.defuzzification(fuzzy_set, method = 'centroid')
            dfz_weights.append(w)
            
        ops_x_weights = [group_local_opinions[i]*dfz_weights[i] for i in range(len(self.other_bettors))]
        
        # self weight to be placed on new opinion calculation - opinion gap of zero
        #self_weight = fuzzy_bc.fuzzification(mfx, 0) 
        
        # dont fuzzify opinion gap 0 for self weight, should just be 1
        self_weight = 1
        dfz_weights.append(self_weight)
        self.temp_group_interaction_log['weights'].append(dfz_weights)
        
        self_update = X_i*self_weight
        ops_x_weights.append(self_update)
        self.temp_group_interaction_log['ops_x_weights'].append(ops_x_weights)
        
        new_xi_opinion = sum(ops_x_weights)/sum(dfz_weights)

        self.temp_group_interaction_log['b1_new_local_op'].append(round(new_xi_opinion, 2))
        
        self.temp_group_interaction_log['b1_op_change'].append(round(new_xi_opinion - X_i, 2))
        
        self.bettor_initiator.set_opinion(new_xi_opinion)

        # append temporary log dict to actual interaction log dict
        # necessary to do this way as conversations vary in length so done this way
        # to correctly keep each row of data together
        for key, value in self.temp_group_interaction_log.items():
            self.interaction_log[key].append(value[0])
            
            
    def ambiguous_opinion(self, exact_opinion, amount):
        
        if amount == 'medium':
            # strongest opinions either poisitive or negative have least ambiguity 
            if 0 <= exact_opinion <= 0.2:
                expressed_opinion = round(max(random.uniform(exact_opinion-0.05, exact_opinion+0.05), 0), 2)
            elif 0.8 <= exact_opinion <= 1:
                expressed_opinion = round(min(random.uniform(exact_opinion-0.05, exact_opinion+0.05), 1), 2)
            # reasonably strong opinions have slightly more ambiguity
            elif (0.2 < exact_opinion <= 0.4) or (0.6 <= exact_opinion < 0.8):
                expressed_opinion = round(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 2)  
            # neutral opinions have greatest ambiguity
            elif 0.4 < exact_opinion < 0.6:
                expressed_opinion = round(random.uniform(exact_opinion-0.15, exact_opinion+0.15), 2)
                
            return expressed_opinion
        
        elif amount == 'low':
            # strongest opinions either poisitive or negative have least ambiguity 
            if 0 <= exact_opinion <= 0.2:
                expressed_opinion = round(max(random.uniform(exact_opinion-0.02, exact_opinion+0.02), 0), 2)
            elif 0.8 <= exact_opinion <= 1:
                expressed_opinion = round(min(random.uniform(exact_opinion-0.02, exact_opinion+0.02), 1), 2)
            # reasonably strong opinions have slightly more ambiguity
            elif (0.2 < exact_opinion <= 0.4) or (0.6 <= exact_opinion < 0.8):
                expressed_opinion = round(random.uniform(exact_opinion-0.05, exact_opinion+0.05), 2)
            # neutral opinions have greatest ambiguity
            elif 0.4 < exact_opinion < 0.6:
                expressed_opinion = round(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 2)
                
            return expressed_opinion
        
        elif amount == 'high':
            # strongest opinions either poisitive or negative have least ambiguity 
            if 0 <= exact_opinion <= 0.2:
                expressed_opinion = round(max(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 0), 2)
            elif 0.8 <= exact_opinion <= 1:
                expressed_opinion = round(min(random.uniform(exact_opinion-0.1, exact_opinion+0.1), 1), 2)
            # reasonably strong opinions have slightly more ambiguity
            elif (0.2 < exact_opinion <= 0.4) or (0.6 <= exact_opinion < 0.8):
                expressed_opinion = round(random.uniform(exact_opinion-0.18, exact_opinion+0.18), 2)  
            # neutral opinions have greatest ambiguity
            elif 0.4 < exact_opinion < 0.6:
                expressed_opinion = round(random.uniform(exact_opinion-0.25, exact_opinion+0.25), 2)
                
            return expressed_opinion
            
       

class OpinionDynamicsPlatform:
    def __init__(self, bettors, model,
                 network_structure, interaction_type, interaction_selection, muddle_opinions):
        self.bettors = bettors
        self.model = model
        self.conversations = []
        self.number_of_conversations = 0
        
        self.network_structure = network_structure
        self.interaction_type = interaction_type
        self.interaction_selection = interaction_selection
        self.muddle_opinions = muddle_opinions

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
        
        # if network structure desired, initialise network and output structure csv 
        if self.network_structure == 'watts_strogatz':
            
            watts_strogatz = WattsStrogatz(len(self.all_opinionated), NUM_NEIGHBOURS, REWIRING_PROB)
            self.network = watts_strogatz.create_network()
            
            # create network structure data frame to output for use in Tableau
            ids = []
            nodes = []
            bettor_types = []
            edges = []
            degrees = []
            
            for i in range(len(self.network.vertex())): 
                ids.append(self.all_opinionated[i].id)
                nodes.append(np.sort(self.network.vertex())[i])
                bettor_types.append(str(self.all_opinionated[i]).lstrip("<betting_agents.Agent_Opinionated_").split().pop(0))
                edges.append(self.network.degree(i))
                degrees.append(self.network.edge(i))
            
                data = {#'id': [bettor.id for bettor in self.bettors],
                        'id': ids,
                        'shuffled_id': nodes,
                        'bettor type': bettor_types,
                        'Number of Neighbours': edges,
                        'Neighbours': degrees}
                
                df = pd.DataFrame(data)
                df.to_csv('data/network_structure.csv',
                          index = False)
        
        pairwise_interaction_log = {'type': [], 
                                    'time': [], 
                                    'length': [],
                                    'bettor1': [], 
                                    'bettor1_id': [],
                                    'b1_local_op': [], 
                                    'b1_uncertainty': [],
                                    'bettor2': [],
                                    'bettor2_id': [],
                                    'deg_of_connection': [],
                                    'b2_local_op': [],
                                    'b2_uncertainty': [],
                                    'b2_expressed_op': [],
                                    'local_op_gap': [],
                                    'weight': [],
                                    'b1_new_local_op': [],
                                    'b1_op_change': [],
                                    'b2_new_local_op': []}     
   
        group_interaction_log = {'type': [], 
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
        
        if self.interaction_type == 'pairwise':
            self.interaction_log = pairwise_interaction_log
        elif self.interaction_type == 'group':
            self.interaction_log = group_interaction_log
        
        
    def initiate_conversations(self, time):
        
        if self.interaction_type == 'pairwise':
        
            if self.network_structure == 'fully_connected':
                
                #for bettor in self.available_influenced_by_opinions:
    
                #while len(self.available_influenced_by_opinions) > 0:
                    
                # while number of available OI bettors is greater than 10% of its start value
                # to always keep some OI bettors available
                while len(self.available_influenced_by_opinions) > 0.1*len(self.all_influenced_by_opinions):
                    
                    #bettor = self.available_influenced_by_opinions[0]
                    
                    # randomly sample from OI bettors not just take first to prevent bias on ID order
                    bettor = random.sample(self.available_influenced_by_opinions, 1)[0]
                    bettor1 = bettor
                    bettor2 = bettor
                    
                    while bettor1 == bettor2:
                        # if not at least one other available bettor, return
                        if len(self.available_opinionated) < 2:
                            return
                        else:
                            bettor2 = random.sample(self.available_opinionated, 1)[0]
                            
                    id = self.number_of_conversations
        
                    Conversation = LocalConversation(id, bettor1, bettor2, time, self.model,
                                                     self.interaction_log, self.muddle_opinions)
        
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
                #while len(self.available_influenced_by_opinions) > 0:
                    
                # while number of available OI bettors is greater than 10% of its start value
                # to always keep some OI bettors available - mainly to test across network method
                while len(self.available_influenced_by_opinions) > 0.1*len(self.all_influenced_by_opinions):
                    
                    #bettor = self.available_influenced_by_opinions[0]
                    
                    # randomly sample from OI bettors not just take first to prevent bias
                    bettor = random.sample(self.available_influenced_by_opinions, 1)[0]
        
                    self.bettor1 = bettor
                    bettor2 = bettor
                    
                    self.bettor1_id = self.all_opinionated.index(self.bettor1)
                    
                    # retrieve neighbour ids based on created network, implying id in list is its node id in network
                    bettor_neighbours_ids = self.network.edge(self.bettor1_id)

                    bettor1_neighbours = []
                    
                    for i in bettor_neighbours_ids:
                        bettor1_neighbours.append(self.all_opinionated[i])
                    
                    self.available_neighbours = [bettor for bettor in bettor1_neighbours if
                                                 bettor.in_conversation == 0]
                    
                    while self.bettor1 == bettor2:
                        
                        if self.interaction_selection == 'direct_neighbours':
                            # if not at least one available neighbour, return
                            if len(self.available_neighbours) < 1:
                                return
                            else:
                                bettor2 = self.select_network_interaction_bettors(self.interaction_type,
                                                                                  self.interaction_selection)
                        elif self.interaction_selection == 'across_network':
                            # if not at least one other available bettor, return
                            if len(self.available_opinionated) < 2:
                                return
                            else:
                                bettor2 = self.select_network_interaction_bettors(self.interaction_type,
                                                                                  self.interaction_selection)

                    id = self.number_of_conversations      

                    Conversation = LocalConversation(id, self.bettor1, bettor2, time, self.model, 
                                                     self.interaction_log, self.muddle_opinions)
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                
                    
        elif self.interaction_type == 'group':
            
            if self.network_structure == 'fully_connected':
                
                #for bettor in self.available_influenced_by_opinions:
                    
                # while number of available OI bettors is greater than 10% of its start value
                # to always keep some OI bettors available
                while len(self.available_influenced_by_opinions) > 0.1*len(self.all_influenced_by_opinions):
                                       
                    #bettor = self.available_influenced_by_opinions[0]
                    
                    # randomly sample from OI bettors not just take first to prevent bias
                    bettor = random.sample(self.available_influenced_by_opinions, 1)[0]
                    bettor1 = bettor
                    
                    # if not at least one other available bettor, return
                    if len(self.available_opinionated) < 2:
                        return
                    else:
                        # number of other bettors to be in group conversation
                        num_bettors_to_select = random.randint(1, min(10, len(self.available_opinionated)))
                        conv_group = random.sample(self.available_opinionated, num_bettors_to_select) # rndomly select given amount of available neighbours
                    
                    # if bettor1 in the group, resample until not in group
                    while bettor1 in conv_group:
                        # redraw num bettors in case is max number of available bettors including bettor 1
                        num_bettors_to_select = random.randint(1, min(10, len(self.available_opinionated)))
                        conv_group = random.sample(self.available_opinionated, num_bettors_to_select)
                        
                            
                    id = self.number_of_conversations
                    
                    Conversation = GroupConversation(id, bettor1, conv_group, time, self.model, 
                                                     self.interaction_log, self.muddle_opinions)                
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
        
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                    
                
            elif self.network_structure == 'watts_strogatz':
                
                #for bettor in self.available_influenced_by_opinions:
                    
                #while len(self.available_influenced_by_opinions) > 0:
                
                # while number of available OI bettors is greater than 10% of its start value
                # to always keep some OI bettors available
                while len(self.available_influenced_by_opinions) > 0.1*len(self.all_influenced_by_opinions):
                    
                    #bettor = self.available_influenced_by_opinions[0]
                    
                    # randomly sample from OI bettors not just take first to prevent bias
                    bettor = random.sample(self.available_influenced_by_opinions, 1)[0]

                    self.bettor1 = bettor
                    
                    self.bettor1_id = self.all_opinionated.index(self.bettor1)
                    
                    bettor_neighbours_ids = self.network.edge(self.bettor1_id)
                          
                    all_neighbours = []
                    
                    for i in bettor_neighbours_ids:
                        all_neighbours.append(self.all_opinionated[i])
                        
                    
                    self.available_neighbours = [bettor for bettor in all_neighbours if
                                                 bettor.in_conversation == 0]   
                    
                    if self.interaction_selection == 'direct_neighbours':
                        # if not at least one available neighbour, return
                        if len(self.available_neighbours) < 1:
                            return
                        else:
                            # number of other bettors to be in group conversation, maximum 10
                            #num_bettors_to_select = random.randint(1, min(10, len(self.available_neighbours)))
                            conv_group = self.select_network_interaction_bettors(self.interaction_type,
                                                                                 self.interaction_selection)

                    elif self.interaction_selection == 'across_network':
                        # if not at least one other available bettor, return
                        if len(self.available_opinionated) < 2:
                            return
                        else:
                            # number of other bettors to be in group conversation
                            #num_bettors_to_select = random.randint(1, min(10, len(self.available_opinionated)))
                            conv_group = self.select_network_interaction_bettors(self.interaction_type,
                                                                                 self.interaction_selection)

                    id = self.number_of_conversations
                    
                    Conversation = GroupConversation(id, self.bettor1, conv_group, time, self.model, 
                                                     self.interaction_log, self.muddle_opinions)
        
                    self.available_influenced_by_opinions = [bettor for bettor in self.all_influenced_by_opinions if
                                                             bettor.in_conversation == 0]
                    
                    self.available_opinionated = [bettor for bettor in self.all_opinionated if
                                                  bettor.in_conversation == 0]
                    
                    self.conversations.append(Conversation)
                    self.number_of_conversations = self.number_of_conversations + 1
                        



    def select_network_interaction_bettors(self, interaction_type, interaction_selection):
        
        self.interaction_type = interaction_type
        self.interaction_selection = interaction_selection
        
        #num_bettors_to_select = num_bettors_to_select
    
        
        if self.interaction_selection == 'direct_neighbours':
            
            for bettor in self.all_opinionated:
                bettor.degree_of_connection = 1
            
            if self.interaction_type == 'pairwise':
                # randomly select one available neighbour
                bettor2 = random.sample(self.available_neighbours, 1)[0]
                return bettor2
            
            elif self.interaction_type == 'group':
                # randomly select given amount of available neighbours
                num_bettors_to_select = random.randint(1, min(10, len(self.available_neighbours)))
                conv_group = np.random.choice(self.available_neighbours, 
                                              num_bettors_to_select,
                                              replace = False)
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
            
            # remove bettor 1 from self.available_opinionated so cant be selected
            self.available_opinionated.remove(self.bettor1)
            
            sample_prob_dist = []
            
            #available_privs = 0
            #available_else = 0
            
            for bettor in self.available_opinionated:
                if bettor.degree_of_connection == 1:
                    # need probabilities to match up with both total pop size and initial neighbour
                    # size to get realistic results
                    # if degree 1 prob too high then in smaller pop will be only degree selected
                    # if degree 1 prob too small then in larger pop, mostly lesser degrees selected
                    
                    sample_prob_dist.append(8/len(self.available_opinionated)) # highest prob
                elif bettor.degree_of_connection == 2:
                    sample_prob_dist.append(4/len(self.available_opinionated))
                elif bettor.degree_of_connection == 3:
                    sample_prob_dist.append(2/len(self.available_opinionated))
                elif bettor.degree_of_connection == 4:
                    sample_prob_dist.append(1/len(self.available_opinionated)) # lowest prob
                    
            #print('available privs: ', available_privs)
            #print('available else: ', available_else)

            # normalise probability distribution so sums to one
            norm_sample_prob_dist = [prob/sum(sample_prob_dist) for prob in sample_prob_dist]
            
            if self.interaction_type == 'pairwise':
                # select one bettor randomly with given probability distribution
                bettor2 = np.random.choice(self.available_opinionated, 1, p = norm_sample_prob_dist)[0]
                return bettor2
            
            elif self.interaction_type == 'group':
                
                def weighted_sample_without_replacement(population, weights, k, rng=random):
                    v = [rng.random() ** (1 / w) for w in weights]
                    order = sorted(range(len(population)), key=lambda i: v[i])
                    return [population[i] for i in order[-k:]]
                
                num_bettors_to_select = random.randint(1, min(10, len(self.available_opinionated)))
                
                bettors = weighted_sample_without_replacement(population = self.available_opinionated, 
                                                              weights = norm_sample_prob_dist,
                                                              k = num_bettors_to_select,
                                                              rng=random)
                return bettors
                
     
    def output_interaction_log(self):
        
        interaction_log_df = pd.DataFrame.from_dict(self.interaction_log, orient='index')
        interaction_log_df = interaction_log_df.transpose() # essential with orient = index or else different lengths
        
        return interaction_log_df
              

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



