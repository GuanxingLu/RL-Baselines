'''
NOTE: This script only support gym 0.26.2
'''
import gym
import readchar
import numpy as np
import argparse
import pickle
 
Push_Left = 0
Push_Right = 1

env_name = 'CartPole-v0'
 
arrow_keys = {
    'a': Push_Left,
    'd': Push_Right
    }

def main(n_trajectories):
    info=f""" 
        First, (select) click the terminal not the game window.
        Now, press keys to control the car.

        keys:
        {arrow_keys}

        Collect human demonstrations for {env_name}
    """
    
    print(info)
    env = gym.make(env_name, render_mode='human')
    trajectories = []

    total_reward=0
    for episode in range(n_trajectories):
        trajectory = []
        step = 0
 
        env.reset() 
        episode_reward=0
        while True: 
            env.render()  
            key = readchar.readkey() 

            if key not in arrow_keys.keys():
                print(f'key {key} is not in {arrow_keys.keys()}, skip demo')
                break 

            action = arrow_keys[key]
            state, reward, done, trunc, info = env.step(action)
            episode_reward += reward

            if done or trunc:
                print('done or truncated')
                break

            trajectory.append((state, action))
            step += 1

        total_reward += episode_reward
        trajectories.append(trajectory)

        print(f'\nEpisode: {episode+1}/{n_trajectories} | reward: {episode_reward}')

    env.close()
 
    average_reward=total_reward // n_trajectories

    print(f'Total trajectory: {n_trajectories} |  Average reward: {average_reward}')
    
    data_path = "expert_data/"+f'human_demos_{n_trajectories}_{average_reward}.pkl'

    with open(data_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print('Expert trajectories saved')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--n", type=int, default=1)
    args = args.parse_args()
    main(args.n)
