import numpy as np
import ENV
import AC
import torch
import rl_utils
from matplotlib import pyplot as plt
from PRINT import Logger
from TEST import model_test
from RANDOMAGENT import RANDOMAGENT, RANDOMAGENT_onehot
from copy import deepcopy

#torch.autograd.set_detect_anomaly(True)
logger = Logger('AC.log')
np.random.seed(1)
torch.manual_seed(0)
lr = 1e-4
#critic_lr = 1e-1
num_episodes = 100
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=4
num_units=1
bs_cover=100
'''config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(100,200),
        'sr':random_uniform_int(500,1000),
        'tp':random_uniform_float(1,3),
        'pfr':random_uniform_int(1000,2000),
        'pf':random_uniform_int(800,2000),
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
        'alpha':2}'''
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(1000,2000),
        'sr':random_uniform_int(20,40),
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
        'alpha':2}

num_subtasks=4
time_base=2
weights=np.ones(8)
weights[0]=1
weights[1]=0
env=ENV.ENVONE(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)

w=(env.num_processors,env.num_processor_attributes-1+env.num_subtasks)
#agent=AC.ActorCritic(w,num_subtasks,actor_lr,critic_lr,gamma,device,clip_grad=1,beta=0,conv=1)
agent=AC.ActorCritic_Double(w,num_subtasks,lr,10,gamma,device,clip_grad=0.1,beta=0.01,conv=1)
return_list=rl_utils.train_on_policy_agent(env,agent,num_episodes)

l1=model_test(env,agent,10,num_subtasks,cycles=10)
print('next_agent##################################################')
r_agent=RANDOMAGENT_onehot(w,num_subtasks)
l2=model_test(env,r_agent,10,num_subtasks,cycles=10)
print(np.array(l1).sum(),np.array(l2).sum())
logger.reset()

'''epoisodes_list=list(range(len(return_list)))
plt.plot(epoisodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on MY_ENV')
plt.show()

mv_return=rl_utils.moving_average(return_list,9)
plt.plot(epoisodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on MY_ENV')
plt.show()'''