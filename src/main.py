#!/usr/bin/python

import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import simulationBitcoin
import simulationEth
import simulationMonero
import networkAnalytics
import TheorySimulationBitcoin
import TheorySimulationEthereum
import TheorySimulationMonero
import networkAnalytics


def main(show_plots):
    t_start = 1
    t_end = 2000
    n_iterations = 100000
    plot_first_x_graphs = 0
    avg_paths_after_n_iterations = [] # [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    # outbound_distributions = ['const8_125', 'const8_inf', 'const13_inf', 'const13_25']
    # make sure that the outbound distribution is consistent with max_outbound_distribution
    outbound_distribution ='const8_125'
    max_outbound_connections = 8
    p_a = 0.95
    compare_growing_network(t_start=t_start, t_end=t_end, n_iterations=n_iterations,
                            plot_first_x_graphs=plot_first_x_graphs,
                            avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                            MAX_OUTBOUND_CONNECTIONS=max_outbound_connections,
                            outbound_distribution=outbound_distribution, p_a=p_a)
    #big_simulation_bitcoin()
    #big_simulation_ethereum()
    #big_simulation_monero()

def compare_growing_network(t_start, t_end, n_iterations, plot_first_x_graphs, avg_paths_after_n_iterations,
                            MAX_OUTBOUND_CONNECTIONS, outbound_distribution, p_a):
    y = list()
    ### BITCOIN ###
    s = TheorySimulationBitcoin.Simulation(simulation_type='bitcoin_protocol', with_evil_nodes=False,
                                           connection_strategy='stand_bc',
                                           initial_connection_filter=False,
                                           outbound_distribution=outbound_distribution,
                                           data={'initial_min': -1, 'initial_max': -1},
                                           MAX_OUTBOUND_CONNECTIONS=MAX_OUTBOUND_CONNECTIONS)
    y.append(s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations,
                   plot_first_x_graphs=plot_first_x_graphs,
                   avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                   MAX_OUTBOUND_CONNECTIONS=MAX_OUTBOUND_CONNECTIONS, numb_nodes=3000, p_a=p_a))

    ### ETHEREUM ###
    s = TheorySimulationEthereum.Simulation(simulation_type='ethereum_protocol', with_evil_nodes=False,
                                            connection_strategy='stand_eth',
                                            initial_connection_filter=False,
                                            outbound_distribution=outbound_distribution,
                                            data={'initial_min': -1, 'initial_max': -1},
                                            MAX_OUTBOUND_CONNECTIONS=MAX_OUTBOUND_CONNECTIONS)
    y.append(s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations,
                   plot_first_x_graphs=plot_first_x_graphs,
                   avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                   MAX_OUTBOUND_CONNECTIONS=MAX_OUTBOUND_CONNECTIONS, numb_nodes=3000, p_a=p_a))

    ### MONERO ###
    s = TheorySimulationMonero.Simulation(simulation_type='monero_protocol', with_evil_nodes=False,
                                          connection_strategy='stand_mon',
                                          initial_connection_filter=False,
                                          outbound_distribution=outbound_distribution,
                                          data={'initial_min': -1, 'initial_max': -1},
                                          MAX_OUTBOUND_CONNECTIONS=MAX_OUTBOUND_CONNECTIONS)
    y.append(s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations,
                   plot_first_x_graphs=plot_first_x_graphs,
                   avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                   MAX_OUTBOUND_CONNECTIONS=MAX_OUTBOUND_CONNECTIONS, numb_nodes=3000, p_a=p_a))


def big_simulation_bitcoin():
    t_start = 1
    t_end = 2000
    n_iterations = 100000
    plot_first_x_graphs = 0
    avg_paths_after_n_iterations = [] # [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    y = list()
    # outbound_distributions = ['const8_125', 'const8_inf', 'const13_inf', 'const13_25']
    # make sure that the outbound distribution is consistent with max_outbound_distribution
    outbound_distribution = 'const8_125'
    max_outbound_connections = 8
    s = simulation.SimulationBitcoin(simulation_type='bitcoin_protocol', with_evil_nodes=False,
                                                           connection_strategy='stand_bc',
                                                           initial_connection_filter=False,
                                                           outbound_distribution=outbound_distribution,
                                                           data={'initial_min': -1, 'initial_max': -1},
                                                           MAX_OUTBOUND_CONNECTIONS=max_outbound_connections)
    y.append(s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations, plot_first_x_graphs=plot_first_x_graphs,
                   avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                   MAX_OUTBOUND_CONNECTIONS=max_outbound_connections, numb_nodes=3000))

def big_simulation_ethereum():
    t_start = 1
    t_end = 2000
    n_iterations = 100000
    plot_first_x_graphs = 0
    avg_paths_after_n_iterations = [] # [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    y = list()
    # outbound_distributions = ['const8_125', 'const8_inf', 'const13_inf', 'const13_25']
    # make sure that the outbound distribution is consistent with max_outbound_distribution
    outbound_distribution = 'const13_25'
    max_outbound_connections = 13
    s = simulationEth.Simulation(simulation_type='ethereum_protocol', with_evil_nodes=False,
                                                            connection_strategy='stand_eth',
                                                            initial_connection_filter=False,
                                                            outbound_distribution=outbound_distribution,
                                                            data={'initial_min': -1, 'initial_max': -1},
                                                            MAX_OUTBOUND_CONNECTIONS=max_outbound_connections)
    y.append(s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations, plot_first_x_graphs=plot_first_x_graphs,
                   avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                   MAX_OUTBOUND_CONNECTIONS=max_outbound_connections, numb_nodes=3000))

def big_simulation_monero():
    t_start = 1
    t_end = 2000
    n_iterations = 100000
    plot_first_x_graphs = 0
    avg_paths_after_n_iterations = [] # [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]
    y = list()
    outbound_distribution = 'const8_inf'
    max_outbound_connections = 8
    s = simulationMonero.Simulation(simulation_type='monero_protocol', with_evil_nodes=False,
                                                          connection_strategy='stand_mon',
                                                          initial_connection_filter=False,
                                                          outbound_distribution=outbound_distribution,
                                                          data={'initial_min': -1, 'initial_max': -1},
                                                          MAX_OUTBOUND_CONNECTIONS=max_outbound_connections)
    y.append(s.run(t_start=t_start, t_end=t_end, n_iterations=n_iterations, plot_first_x_graphs=plot_first_x_graphs,
                   avg_paths_after_n_iterations=avg_paths_after_n_iterations,
                   MAX_OUTBOUND_CONNECTIONS=max_outbound_connections, numb_nodes=3000))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='hoi Lukas')
    parser.add_argument('-display', default=True, help='Can show stuff on the display?')
    args = parser.parse_args()
    main2(bool(args.display))
