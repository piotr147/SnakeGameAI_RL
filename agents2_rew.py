import torch
import random
import numpy as np
from collections import deque
from snake_2_rew import SnakeGameAI2,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agents:
    def __init__(self, rewarders, number_of_snakes = 2):
        self.n_snakes = number_of_snakes
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.rewarders = rewarders

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
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    # rewarders = [Rewarder1(), Rewarder1(opponent_took_food=0)]
    # agent = Agents(rewarders, 2)
    rewarders = [Rewarder1()]
    agent = Agents(rewarders, 1)
    game = SnakeGameAI2(n = agent.n_snakes)
    records = [0 for _ in range(agent.n_snakes)]
    while True:
        game_before = game.create_copy()
        # Get Old state
        states_old = [agent.get_state(game, i) for i in range(agent.n_snakes)]

        # get move
        final_moves = [agent.get_action(states_old[i], i) for i in range(agent.n_snakes)]

        # perform move and get new state
        done, scores = game.play_step(final_moves)
        states_new = [agent.get_state(game, i) for i in range(agent.n_snakes)]
        rewards = [rewarders[i].calculate_reward(game_before, game, i) for i in range(agent.n_snakes)]

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

            # plot_scores.append(score)
            # total_score+=score
            # mean_score = total_score / agent.n_game
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores,plot_mean_scores)


class Rewarder1:
    def __init__(self, food_taken = 100, death = -100, iterations_exceeded = -100, closer_to_food = 2, further_from_food = -2, opponent_took_food = -50):
        self.food_taken = food_taken
        self.death = death
        self.iterations_exceeded = iterations_exceeded
        self.closer_to_food = closer_to_food
        self.further_from_food = further_from_food
        self.opponent_took_food = opponent_took_food

    def calculate_reward(self, before, after, snake_id):
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

        return reward



if(__name__=="__main__"):
    train()