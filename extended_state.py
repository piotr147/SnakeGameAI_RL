from ast import AsyncFunctionDef
from logging import setLogRecordFactory
from tarfile import BLOCKSIZE
import torch
import random
import numpy as np
import csv
from collections import deque
from game_ai import SnakeGameAI2,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, rewarder, model_name, load_from_model = ''):
        self.rewarder = rewarder
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11,256,3, model_name=model_name, load_from_model = load_from_model)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

class AgentsSupervisor:
    def __init__(self, agents):
        self.n_snakes = len(agents)
        self.agents = agents
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate

        self.memories = []
        self.models = []
        self.trainers = []

        for i in range(self.n_snakes):
            self.memories.append(deque(maxlen=MAX_MEMORY)) # popleft()

    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #
    # direction left, direction right,
    # direction up, direction down
    #
    # food left,food right,
    # food up, food down]
    def get_state(self,game,snake_id):
        head = game.snakes[snake_id][0]
        #   ll  l       ls
        #snakee head    s
        #   rr  r       sr
        ####################
        #   lu  u       ur
        #   l   head    r
        #   ld  d       rd
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)
        point_lu = Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_ur = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE)
        point_rd = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)
        point_ld = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)

        dir_l = game.directions[snake_id] == Direction.LEFT
        dir_r = game.directions[snake_id] == Direction.RIGHT
        dir_u = game.directions[snake_id] == Direction.UP
        dir_d = game.directions[snake_id] == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(snake_id, point_u))or
            (dir_d and game.is_collision(snake_id, point_d))or
            (dir_l and game.is_collision(snake_id, point_l))or
            (dir_r and game.is_collision(snake_id, point_r)),

            # Danger right
            (dir_u and game.is_collision(snake_id, point_r))or
            (dir_d and game.is_collision(snake_id, point_l))or
            (dir_l and game.is_collision(snake_id, point_u))or
            (dir_r and game.is_collision(snake_id, point_d)),

            #Danger Left
            (dir_d and game.is_collision(snake_id, point_r))or
            (dir_u and game.is_collision(snake_id, point_l))or
            (dir_r and game.is_collision(snake_id, point_u))or
            (dir_l and game.is_collision(snake_id, point_d)),

            #Danger LeftLeft
            (dir_d and game.is_collision(snake_id, point_ur))or
            (dir_u and game.is_collision(snake_id, point_ld))or
            (dir_r and game.is_collision(snake_id, point_lu))or
            (dir_l and game.is_collision(snake_id, point_rd)),

            #Danger LeftStraigth
            (dir_d and game.is_collision(snake_id, point_rd))or
            (dir_u and game.is_collision(snake_id, point_lu))or
            (dir_r and game.is_collision(snake_id, point_ur))or
            (dir_l and game.is_collision(snake_id, point_ld)),

            #Danger StraigthRight
            (dir_d and game.is_collision(snake_id, point_ld))or
            (dir_u and game.is_collision(snake_id, point_ur))or
            (dir_r and game.is_collision(snake_id, point_rd))or
            (dir_l and game.is_collision(snake_id, point_lu)),

            #Danger RightRight
            (dir_d and game.is_collision(snake_id, point_lu))or
            (dir_u and game.is_collision(snake_id, point_rd))or
            (dir_r and game.is_collision(snake_id, point_ld))or
            (dir_l and game.is_collision(snake_id, point_ur)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.heads[snake_id].x, # food is in left
            game.food.x > game.heads[snake_id].x, # food is in right
            game.food.y < game.heads[snake_id].y, # food is up
            game.food.y > game.heads[snake_id].y  # food is down
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done,snake_id):
        self.agents[snake_id].memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self, snake_id):
        if (len(self.agents[snake_id].memory) > BATCH_SIZE):
            mini_sample = random.sample(self.agents[snake_id].memory,BATCH_SIZE)
        else:
            mini_sample = self.agents[snake_id].memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.agents[snake_id].trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done,snake_id):
        self.agents[snake_id].trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state,snake_id):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 1000 - self.n_game
        if self.epsilon <= 0:
            self.epsilon = 2
        final_move = [0,0,0]
        if(random.randint(0,1000)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float).cpu()
            prediction = self.agents[snake_id].model(state0).cpu() # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

def train(snake_type=1, iterations=1000, filename="default_result.csv", agents = [], random_rounds=500):

    supervisor = AgentsSupervisor(agents, random_rounds=random_rounds)

    game = SnakeGameAI2(n = supervisor.n_snakes)
    records = [0 for _ in range(supervisor.n_snakes)]
    csv_list = []

    while supervisor.n_game < iterations:
        game_before = game.create_copy()
        # Get Old state
        states_old = [supervisor.get_state(game, i) for i in range(supervisor.n_snakes)]

        # get move
        final_moves = [supervisor.get_action(states_old[i], i) for i in range(supervisor.n_snakes)]

        # perform move and get new state
        done, scores = game.play_step(final_moves)
        states_new = [supervisor.get_state(game, i) for i in range(supervisor.n_snakes)]
        rewards = [supervisor.agents[i].rewarder.calculate_reward(supervisor.memories[i], game_before, game, i) for i in range(supervisor.n_snakes)]

        # train short memory
        for i in range(supervisor.n_snakes):
            supervisor.train_short_memory(states_old[i],final_moves[i],rewards[i],states_new[i],done,i)

        #remember
        for i in range(supervisor.n_snakes):
            supervisor.remember(states_old[i],final_moves[i],rewards[i],states_new[i],done,i)

        if done:
            # Train long memory,plot result
            game.reset()
            supervisor.n_game += 1
            text = 'Game: ' + str(supervisor.n_game)
            for i in range(supervisor.n_snakes):
                supervisor.train_long_memory(i)
                if(scores[i] > records[i]): # new High score
                    records[i] = scores[i]
                    supervisor.agents[i].model.save()

                text = text + ', Score ' + str(i) + ': ' + str(scores[i]).zfill(3) + ', Record ' + str(i) + ': '+ str(records[i]).zfill(3)
            csv_list.append([supervisor.n_game, scores[0]])

            print(text)

    with open(filename,'w', newline='') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(["Game num","result"])
        csv_out.writerows(csv_list)


class Rewarder:
    def __init__(self, food_taken = 100, death = -100, iterations_exceeded = -100, closer_to_food = 2,
    further_from_food = -2, opponent_took_food = 0, cycle_found = 0):
        self.food_taken = food_taken
        self.death = death
        self.iterations_exceeded = iterations_exceeded
        self.closer_to_food = closer_to_food
        self.further_from_food = further_from_food
        self.opponent_took_food = opponent_took_food
        self.cycle_found = cycle_found

    def calculate_reward(self, memories, before, after, snake_id):
        reward = 0

        if(before.dist_to_food(snake_id) > after.dist_to_food(snake_id)):
            reward += self.closer_to_food
        else:
            reward += self.further_from_food

        if(before.scores[snake_id] < after.scores[snake_id]):
            reward += self.food_taken

        if(any(before.scores[i] < after.scores[i] and i != snake_id for i in range(after.n))):
            reward += self.opponent_took_food

        if(after.is_collision(snake_id)):
            reward += self.death

        if(after.frame_iteration > 100*sum(len(sn) for sn in after.snakes) // after.n):
            reward += self.iterations_exceeded

        if(not self._find_cycles(memories, 2, 4) and not self._find_cycles(memories, 3, 6)):
            if(self._find_cycles(memories, 4, 8) or self._find_cycles(memories, 6, 12) or self._find_cycles(memories, 8, 16)):
                reward += self.cycle_found

        return reward

    def _find_cycles(self, memories, cycle_length, history_depth):
        if(len(memories) < history_depth):
            return False
        sequences = []
        new_seq = []
        for i in range(history_depth):
            new_seq.append(memories[len(memories) - 1 - i][1])
            if len(new_seq) == cycle_length:
                sequences.append(new_seq)
                new_seq = []

        for i in range(len(sequences) - 1):
            if(not self._compare_sequences(sequences[i], sequences[i+1])):
                return False
        return True

    def _compare_sequences(self, seq1, seq2):
        if(len(seq1) != len(seq2)):
            return False
        for i in range(len(seq1)):
            for j in range(len(seq1[i])):
                if(seq1[i][j] != seq2[i][j]):
                    return False
        return True

if(__name__=="__main__"):
    train()