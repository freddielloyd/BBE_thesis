import sys, math, threading, time, queue, random, csv, config
from copy import deepcopy
from system_constants import *
from betting_agents import *
from race_simulator import Simulator
from ex_ante_odds_generator import *
from exchange import Exchange
from message_protocols import *
from session_stats import *

def main():
    # Simulation attributes
    currentSimulation = 0
    numOfSimulations = 15
    ####################

    # set things up
    # have while loop for running multiple races
    # within loop instantiate competitors into list
    # run simulation and matching engine
    avgtime = 0
    faveWinPercentage = 0

    horseByOdds = []
    better = []
    worse = []
    same = []
    for i in range(8): # should match number of competitors
        horseByOdds.append(0)
        better.append(0)
        worse.append(0)
        same.append(0)



    while currentSimulation < numOfSimulations:
        simulationId = "Simulation: " + str(currentSimulation)

        # Create race event data
        race = Simulator(8) # argument is number of competitors
        compPool = deepcopy(race.competitors)
        raceAttributes = deepcopy(race.race_attributes)
        raceFilename = str(currentSimulation)
        s = time.time()

        createExAnteOdds(compPool, raceAttributes)

        race.run(raceFilename)
        e = time.time()
        avgtime = avgtime + (e-s)

        exAnteOdds = getExAnteOdds(38) # argument is agent_bettor id  so i predicted odds from agent 38
        # minOdds = min(exAnteOdds)
        # c = exAnteOdds.index(minOdds)
        # print(exAnteOdds)
        
        # forecasted odds by agent 38
        print('forecasted odds by agent 38: ', exAnteOdds)

        # index of odds, first is best odds ie most predicted to win
        exAnteOdds = sorted(range(len(exAnteOdds)), key=lambda k: exAnteOdds[k])
        
        # order of competitors as predicted
        print('ordered c.id predicted finish by agent 38: ', exAnteOdds)
        
        print('actual race order finish: ', race.finished)


        for i in range(len(exAnteOdds)):
            # competitor id, predicted finish by agent, actual finish
            #print(str(exAnteOdds[i]) + " : " + str(race.finished.index(exAnteOdds[i])) +  " : " + str(i))
            print(str('comp id: ' + str(exAnteOdds[i])) + ", actual finish: " + str(race.finished.index(exAnteOdds[i])) +  ", pred finish: " + str(i))
            if race.finished.index(exAnteOdds[i]) < i:
                better[i] += 1
                print("bettor")
            elif race.finished.index(exAnteOdds[i]) > i:
                worse[i] += 1
                print("worse")
            elif race.finished.index(exAnteOdds[i]) == i:
                same[i] += 1
                print("same")
            if race.winner == exAnteOdds[i]:
                horseByOdds[i] += 1


        #
        # if c == race.winner:
        #     faveWinPercentage += 1



        currentSimulation = currentSimulation + 1
        print()

    #print(avgtime / currentSimulation)
    print('number of times predicted winner position actually finishes: ', horseByOdds)
    
    for i in range(len(horseByOdds)):
        horseByOdds[i] /= numOfSimulations
        better[i] /= numOfSimulations
        worse[i] /= numOfSimulations
        same[i] /= numOfSimulations

    print('proportions prediction actually wins: ', horseByOdds)
    print('proportion horses did better than predictions: ', better)
    print('proportion horses did worse than predictions: ', worse)
    print('proportion horses did same than predictions: ', same)





if __name__ == "__main__":
    main()
