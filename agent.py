# CODIGO UTILIZADO EXTRAIDO DE: https://github.com/python-engineer/snake-ai-pytorch
from collections import deque
from game import SGNN, Direction, Point
from model import Linear_Qnet, QTrainer
import numpy as np
from ploter import plot
import random
import torch

# CONSTANTES UTILIZADAS EN EL JUEGO

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001  # ! probar con otro lr original 0.001
GAMMA = 0.9  # ! probar con valores entre 0.8 y 0.9
HIDDE_NET_SIZE = 256  # ! probar con otro numero del centro

# Se define la funcion para entrenar al agente

def train():
    scores_to_plot = []
    plot_mean = []
    total_score = 0
    max_score = 0
    agent = Agent()
    game = SGNN()
    while True:
        old_state = agent.get_state(game)
        finale_move = agent.get_action(old_state)
        reward, game_over, score = game.play_step(finale_move)
        new_state = agent.get_state(game)
        agent.train_short_memory(
            old_state, finale_move, reward, new_state, game_over)
        agent.remember(old_state, finale_move, reward, new_state, game_over)

        if game_over:
            game.reset()
            agent.total_games += 1
            agent.train_long_memory()

            if score > max_score:
                max_score = score
                agent.model.save()

            print('Game: ', agent.total_games,
                  ' Score: ', score, ' Best: ', max_score)

            scores_to_plot.append(score)
            total_score += score
            plot_mean.append(total_score / agent.total_games)
            plot(scores_to_plot, plot_mean)


class Agent:
    def __init__(self):
        self.total_games = 0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, HIDDE_NET_SIZE, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

    def get_state(self, game):
        head = game.snake[0]
        left_point = Point(head.x - 20, head.y)
        right_point = Point(head.x + 20, head.y)
        upper_point = Point(head.x, head.y - 20)
        down_point = Point(head.x, head.y + 20)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            (direction_right and game.is_collision(right_point)) or
            (direction_left and game.is_collision(left_point)) or
            (direction_up and game.is_collision(upper_point)) or
            (direction_down and game.is_collision(down_point)),

            (direction_up and game.is_collision(right_point)) or
            (direction_down and game.is_collision(left_point)) or
            (direction_left and game.is_collision(upper_point)) or
            (direction_right and game.is_collision(down_point)),

            (direction_down and game.is_collision(right_point)) or
            (direction_up and game.is_collision(left_point)) or
            (direction_right and game.is_collision(upper_point)) or
            (direction_left and game.is_collision(down_point)),

            direction_left,
            direction_right,
            direction_up,
            direction_down,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards,
                                next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.total_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_initial = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_initial)
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move


if __name__ == '__main__':
    train()
