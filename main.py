from GameEnv import Car, RacingEnv
from dqn import DDQNAgent
import pygame
import numpy as np

N_GAMES = 100000
REPLACE_TARGET = 50 

car = Car(70, 225, 90, 1, 10, 20, (255, 120, 0), 3)
env = RacingEnv((700, 450))

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0, epsilon_end=0.10, 
                       epsilon_dec=0.9995, replace_target=REPLACE_TARGET, batch_size=512, input_dims=10)
ddqn_scores = []
eps_history = []

#uncomment to load trained model
#ddqn_agent.load_model_()





def main():
    
    
    for e in range(N_GAMES):
        score = 0
        
        counter = 0
        
        env.reset(car)
        done = False
        action = -1
        step = 0
        observation_, reward, done = env.step(car, 0)
        observation = np.array(observation_)
        done2 = False
        
        while not done: 
                
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = env.step(car, action)
            done2 = env.draw(car)
            score += reward
            reward += car.speed
            
            if reward == 0: 
                counter += 1
                
                if counter > 500:
                    done = True
            else: 
                counter = 0
            
            if done2:
                return
            
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            
            ddqn_agent.learn()
            
        
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])
        
        
        print("Episode: ", e, " score: ", score, " with average: ", avg_score, " and epsilon: ", ddqn_agent.epsilon)
        
        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()
            

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
  
    env.quit()

main()