import numpy as np

def model_test(env,agent,num_episodes,num_subtasks,cycles=10):
    return_list = []
    for i in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = 0
        while not done:
            print('sub_loc:\n{}'.format(state[0][0,0,:,-num_subtasks:]))
            action = agent.take_action(state)
            print('action0:{}\naction1:{}'.format(action[0],action[1]))
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        if (i+1)%cycles==0:
            print('episode:{}, reward:{}'.format(i+1,np.mean(return_list[-cycles])))
    return return_list