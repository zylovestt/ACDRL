import numpy as np
import ENV
import torch
import rl_utils
from matplotlib import pyplot as plt
import PPO

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=3
num_units=3
bs_cover=1000
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(100,200),
        'sr':random_uniform_int(500,1000),
        'tp':random_uniform_float(1,3),
        'pfr':random_uniform_int(5000,10000),
        'pf':random_uniform_int(2000,5000),
        'plr':unit_loc(0,bs_cover),
        'pl':random_loc(0,bs_cover//10),
        'pd':random_uniform_float(2,6),
        'ps':random_uniform_float(60,100),
        'pbr':random_uniform_int(2000,5000),
        'pb':random_uniform_int(2000,4000),
        'ppr':random_uniform_int(200,300),
        'pp':random_uniform_int(100,200),
        'pg':random_uniform_int(20,40),
        'pcr':random_uniform_float(50,300),
        'pc':random_uniform_float(10,100),
        'whitenoise':1,
        'alpha':2
}

num_subtasks=3
time_base=100

actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 1000
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env=ENV.ENVONE(time_base,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)

torch.manual_seed(0)
w=env.reset()[0].shape[-2:]

agent = PPO.PPO(w,num_subtasks, actor_lr, critic_lr,  gamma, device,lmbda,epochs, eps,conv=1)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on my_env')
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on my_env')
plt.show()