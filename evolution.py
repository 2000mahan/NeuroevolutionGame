import copy
import numpy as np
import pickle
import random

from player import Player

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        # f.close()

    def max_min_average_fitness_save(self, players, average):
        f = open("result.txt", "a")
        result = str(players[0].fitness) + " " + str(average) + " " + str(players[299].fitness) + "\n"
        f.write(result)


    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)
        # TODO (Additional: Learning curve)

        average = 0
        for p in players:
            average += p.fitness

        average = average/300

        # this sorting is for plotting the result
        players.sort(key=lambda x: x.fitness, reverse=True)
        # this line is used for saving the max and min and average fitness of players
        self.max_min_average_fitness_save(players, average)

        fitness_list = list()
        for p in players:
            fitness_list.append(p.fitness ** 2)

        players = random.choices(players, weights=fitness_list, k=num_players)

        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            children_list = []
            fitness_list = list()
            for p in prev_players:
                fitness_list.append(p.fitness ** 2)

            prev_players = random.choices(prev_players, weights=fitness_list, k=num_players)
            half = int(len(prev_players) / 2) - 1
            for i in range(half):
                # c1 is child number 1 and c2 is child number 2
                c1, c2 = self.crossover_operator(prev_players[i * 2], prev_players[i * 2 + 1])
                c1 = self.mutation_operator(c1)
                c2 = self.mutation_operator(c2)
                children_list.append(c1)
                children_list.append(c2)
            new_generated_players = children_list
            return new_generated_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    # single point crossover
    def crossover_operator(self, p1, p2):
        # p1 is the first parent and p2 is the second parent
        p1_w1 = p1.nn.w1
        p1_w2 = p1.nn.w2
        p1_b1 = p1.nn.b1
        p1_b2 = p1.nn.b2

        p2_w1 = p2.nn.w1
        p2_w2 = p2.nn.w2
        p2_b1 = p2.nn.b1
        p2_b2 = p2.nn.b2

        # c1 is child number 1 and c2 is child number 2
        c1 = self.clone_player(p1)
        c2 = self.clone_player(p2)

        counter = 0

        for a in p1_w1:
            length_a = int(len(a))
            half_of_length_a = int(len(a) / 2)
            c1.nn.w1[counter] = np.append(a[0:half_of_length_a], p2_w1[counter][half_of_length_a:length_a])
            c2.nn.w1[counter] = np.append(p2_w1[counter][0:half_of_length_a], a[half_of_length_a:length_a])
            counter = counter + 1

        counter = 0

        for a in p1_w2:
            length_a = int(len(a))
            half_of_length_a = int(len(a) / 2)
            c1.nn.w2[counter] = np.append(a[0:half_of_length_a], p2_w2[counter][half_of_length_a:length_a])
            c2.nn.w2[counter] = np.append(p2_w2[counter][0:half_of_length_a], a[half_of_length_a:length_a])
            counter = counter + 1

        length_p1_b1 = int(len(p1_b1))
        half_of_length_p1_b1 = int(len(p1_b1) / 2)
        c1.nn.b1[0:half_of_length_p1_b1] = p1_b1[0:half_of_length_p1_b1]
        c1.nn.b1[half_of_length_p1_b1:length_p1_b1] = p2_b1[half_of_length_p1_b1:length_p1_b1]
        c2.nn.b1[0:half_of_length_p1_b1] = p2_b1[0:half_of_length_p1_b1]
        c2.nn.b1[half_of_length_p1_b1:length_p1_b1] = p1_b1[half_of_length_p1_b1:length_p1_b1]

        length_p1_b2 = int(len(p1_b2))
        half_of_length_p1_b2 = int((len(p1_b2) / 2))
        c1.nn.b2[0:half_of_length_p1_b2] = p1_b2[0:half_of_length_p1_b2]
        c1.nn.b2[half_of_length_p1_b2:length_p1_b2] = p2_b2[half_of_length_p1_b2:length_p1_b2]
        c2.nn.b2[0:half_of_length_p1_b2] = p2_b2[0:half_of_length_p1_b2]
        c2.nn.b2[half_of_length_p1_b2:length_p1_b2] = p1_b2[half_of_length_p1_b2:length_p1_b2]

        return c1, c2

    def mutation_operator(self, c):
        # c is the creature to operate mutation on
        c_w1 = c.nn.w1
        c_w2 = c.nn.w2
        c_b1 = c.nn.b1
        c_b2 = c.nn.b2

        # nc is the cloned creature(new creature)
        nc = self.clone_player(c)
        mutate_prob = 0.3
        if np.random.uniform(0, 1, 1)[0] < mutate_prob:
            nc.nn.w1 = c_w1 + np.random.randn(10, 6)
        if np.random.uniform(0, 1, 1)[0] < mutate_prob:
            nc.nn.w2 = c_w2 + np.random.randn(1, 10)
        if np.random.uniform(0, 1, 1)[0] < mutate_prob:
            nc.nn.b1 = c_b1 + np.random.randn(10, 1)
        if np.random.uniform(0, 1, 1)[0] < mutate_prob:
            nc.nn.b2 = c_b2 + np.random.randn(1, 1)

        return nc
