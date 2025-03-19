from GameEnv import RacingEnv, Car
import pygame
import numpy as np

from dqn import DDQNAgent

from collections import deque
import random, math



TOTAL_GAMETIME = 10000
N_EPISODES = 10000
REPLACE_TARGET = 10

car = Car(70, 225, 90, 1, 10, 20, (255, 120, 0), 3)
game = RacingEnv((700, 450))


GameTime = 0 
GameHistory = []
renderFlag = True

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target=REPLACE_TARGET, batch_size=64, input_dims=10,fname='DeepQCar/ddqn_model2.h5')

ddqn_agent.load_model_()
ddqn_agent.update_network_parameters()

ddqn_scores = []
eps_history = []


def run():
    #scores = deque(maxlen=100)

    for e in range(N_EPISODES):
        #reset env 
        game.reset(car)

        done = False
        



        #first step
        observation_, reward, done = game.step(car, 0)
        observation = np.array(observation_)

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
                    return

            #new
            game.draw(car)
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(car, action)
            observation_ = np.array(observation_)

            

            observation = observation_




run()        