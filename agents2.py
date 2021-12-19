import torch
import random
import numpy as np
from collections import deque
from snake_2 import SnakeGameAI2,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agents:
    def __init__(self, number_of_snakes = 2):
        self.n_snakes = number_of_snakes
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate

        self.memories = []
        self.models = []
        self.trainers = []

        for i in range(self.n_snakes):
            self.memories.append(deque(maxlen=MAX_MEMORY)) # popleft()
            self.models.append(Linear_QNet(11,256,3))
            self.trainers.append(QTrainer(self.models[i],lr=LR,gamma=self.gamma))
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)
        # self.model.to('cuda')
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)
        # TODO: model,trainer

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
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

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
        self.memories[snake_id].append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self, snake_id):
        if (len(self.memories[snake_id]) > BATCH_SIZE):
            mini_sample = random.sample(self.memories[snake_id],BATCH_SIZE)
        else:
            mini_sample = self.memories[snake_id]
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainers[snake_id].train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done,snake_id):
        self.trainers[snake_id].train_step(state,action,reward,next_state,done)

    def get_action(self,state,snake_id):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        if self.epsilon <= 0:
            self.epsilon = 2
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float).cpu()
            prediction = self.models[snake_id](state0).cpu() # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

def train():
    agent = Agents()
    game = SnakeGameAI2(n = agent.n_snakes)
    records = [0 for _ in range(agent.n_snakes)]
    while True:
        # Get Old state
        states_old = [agent.get_state(game, i) for i in range(agent.n_snakes)]

        # get move
        final_moves = [agent.get_action(states_old[i], i) for i in range(agent.n_snakes)]

        # perform move and get new state
        rewards, done, scores = game.play_step(final_moves)
        states_new = [agent.get_state(game, i) for i in range(agent.n_snakes)]

        # train short memory
        for i in range(agent.n_snakes):
            agent.train_short_memory(states_old[i],final_moves[i],rewards[i],states_new[i],done,i)

        #remember
        for i in range(agent.n_snakes):
            agent.remember(states_old[i],final_moves[i],rewards[i],states_new[i],done,i)

        if done:
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            text = 'Game: ' + str(agent.n_game)
            for i in range(agent.n_snakes):
                agent.train_long_memory(i)
                if(scores[i] > records[i]): # new High score
                    records[i] = scores[i]
                    agent.models[i].save()
                text = text + ', Score ' + str(i) + ': ' + str(scores[i]) + ', Record ' + str(i) + ': '+ str(records[i]) + ', Rewards ' + str(i) + ': '+ str(rewards[i])
            print(text)


if(__name__=="__main__"):
    train()