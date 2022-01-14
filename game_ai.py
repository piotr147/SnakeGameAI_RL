import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
pygame.init()
import copy
font = pygame.font.SysFont('arial', 25)

# Reset
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point','x , y')

BLOCK_SIZE=20
SPEED = 40
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
PRIMARY_COLS = [(0,0,255), (0,255,0), (255,0,0)]
SECONDARY_COLS = [(0,100,255),(100,255,0), (255,0,100)]
WALLS_COL = (255, 255, 0)
BLACK = (0,0,0)

class SnakeGameAI2:
    def __init__(self,w=640,h=480,n=2):
        self.n = n
        self.w=w
        self.h=h
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        #init game state
        self.reset()
    def reset(self):
        self.directions = [Direction.RIGHT for _ in range(self.n)]
        self.heads = []
        self.snakes = []
        self.scores = [0 for _ in range(self.n)]
        w_diff = self.w // BLOCK_SIZE // (self.n + 1)
        h_diff = self.h // BLOCK_SIZE // (self.n + 1)
        self.walls = [Point(15 * BLOCK_SIZE, i * BLOCK_SIZE) for i in range(5, 20)]
        self.walls += ([Point(16 * BLOCK_SIZE, i * BLOCK_SIZE) for i in range(5, 20)])
        self.walls += ([Point(8 * BLOCK_SIZE, i * BLOCK_SIZE) for i in range(0, 8)])
        self.walls += ([Point(8 * BLOCK_SIZE, i * BLOCK_SIZE) for i in range(16, 24)])
        self.walls += ([Point(23 * BLOCK_SIZE, i * BLOCK_SIZE) for i in range(0, 8)])
        self.walls += ([Point(23 * BLOCK_SIZE, i * BLOCK_SIZE) for i in range(16, 24)])

        for i in range(self.n):
            self.heads.append(Point((w_diff + i * w_diff) * BLOCK_SIZE, (h_diff + i * h_diff) * BLOCK_SIZE))
            self.snakes.append([self.heads[i],
                        Point(self.heads[i].x-BLOCK_SIZE,self.heads[i].y),
                        Point(self.heads[i].x-(2*BLOCK_SIZE),self.heads[i].y)])

        self.food = None
        self._place__food()
        self.frame_iteration = 0


    def _place__food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if(any(self.food in sn for sn in self.snakes)) or (any(self.food in w for w in self.walls)):
            self._place__food()


    def play_step(self, actions):
        self.frame_iteration+=1

        # 1. Collect the user input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()

        # 2. Move
        self._move(actions)
        for i in range(len(self.snakes)):
            self.snakes[i].insert(0,self.heads[i])

        # 3. Check if game Over
        game_over = False

        for i in range(len(self.snakes)):
            if(self.is_collision(i)):
                game_over=True

        if(self.frame_iteration > 100*sum(len(sn) for sn in self.snakes) // self.n):
            game_over=True

        if(game_over):
            game_over=True
            return game_over,self.scores

        # 4. Place new Food or just move
        if(any(self.food == hd for hd in self.heads)):
            scored_ind = [i for i in range(len(self.snakes)) if self.food == self.heads[i]][0]
            self.scores[scored_ind] = self.scores[scored_ind] + 1
            self._place__food()
            for i in range(len(self.snakes)):
                if i == scored_ind:
                    continue
                self.snakes[i].pop()
        else:
            for i in range(len(self.snakes)):
                self.snakes[i].pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game Over and Display Score

        return game_over,self.scores

    def _update_ui(self):
        self.display.fill(BLACK)
        for i in range(len(self.snakes)):
            snake = self.snakes[i]
            for pt in snake:
                pygame.draw.rect(self.display,PRIMARY_COLS[i%len(PRIMARY_COLS)],pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
                pygame.draw.rect(self.display,SECONDARY_COLS[i%len(SECONDARY_COLS)],pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        for w in self.walls:
            pygame.draw.rect(self.display,WALLS_COL,pygame.Rect(w.x,w.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: "+str(self.scores),True,WHITE)
        text = ""
        for i in range(len(self.snakes)):
            text = text + "Score " + str(i) + ":" + str(self.scores[i])
        text = font.render(text,True,WHITE)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(self,actions):
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn
        # [0,0,1] -> Left Turn

        for i in range(len(self.snakes)):

            clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
            idx = clock_wise.index(self.directions[i])
            if np.array_equal(actions[i],[1,0,0]):
                new_dir = clock_wise[idx]
            elif np.array_equal(actions[i],[0,1,0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx] # right Turn
            else:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx] # Left Turn
            self.directions[i] = new_dir

            x = self.heads[i].x
            y = self.heads[i].y
            if(self.directions[i] == Direction.RIGHT):
                x+=BLOCK_SIZE
            elif(self.directions[i] == Direction.LEFT):
                x-=BLOCK_SIZE
            elif(self.directions[i] == Direction.DOWN):
                y+=BLOCK_SIZE
            elif(self.directions[i] == Direction.UP):
                y-=BLOCK_SIZE
            self.heads[i] = Point(x,y)

    def dist_to_food(self, snake_num):
        return abs(self.heads[snake_num].x - self.food.x) + abs(self.heads[snake_num].y - self.food.y)

    def is_collision(self, snake_num, pt = None):
        if(pt is None):
            pt = self.heads[snake_num]
        #hit boundary
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.snakes[snake_num][1:]):
            return True
        for i in range(len(self.snakes)):
            if i == snake_num:
                continue
            if(pt in self.snakes[i][0:]):
                return True
        #walls:
        if pt in self.walls:
            return True
        return False

    def create_copy(self):
        game_copy = SnakeGameAI2(self.w, self.h, self.n)
        game_copy.directions = copy.deepcopy(self.directions)
        game_copy.heads = copy.deepcopy(self.heads)
        game_copy.snakes = copy.deepcopy(self.snakes)
        game_copy.scores = copy.deepcopy(self.scores)
        game_copy.food = copy.deepcopy(self.food)
        game_copy.frame_iteration = copy.deepcopy(self.frame_iteration)

        return game_copy
