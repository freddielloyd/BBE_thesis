# BBE_thesis

This thesis describes an extension of the opinion-integrated agent-based betting exchange simulation
platform Bristol Betting Exchange (BBE). BBE is a synthetic data generator of an in-play market on a
betting exchange, providing a means to generate sufficient data necessary for the discovery of profitable
trading strategies using machine learning techniques. This work considers the design and implementation
of several features intended to improve the realism of the BBE platform by modifying the interactions that
occur between agent-bettors. A recently introduced approach to modelling opinion dynamics has been
implemented, attempting to mathematically capture the observation that humans do not communicate
in exact numbers as in traditional opinion dynamics models, but in language that can cause ambiguity in
how opinions are communicated. This method builds upon the Bounded Confidence (opinion dynamics
model by using utilising a fuzzy logic system for the determination of agent-specific interaction weights.
In addition to exploring the effects of this model within BBE, we introduce a method that introduces
varying degrees of ambiguity into interactions before they occur. To further increase the realism of the in-
teractions between agents, a network structure is implemented into the population via the Watts-Strogatz
method. The impact of using such a structure in selecting the agent-bettors for an interaction is explored,
from a completely random process to the direct neighbours of an agent. A novel method for selecting
participants based on their degree of connection to the bettor initiating the interaction is then explored
to achieve a medium between these two extremes. Finally, group-level versions of the original Bounded
Confidence and new Fuzzy Bounded Confidence model are implemented to compare the effects of larger
group interactions with their pairwise counterparts.
