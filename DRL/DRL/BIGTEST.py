import numpy as np
import ENV_AGENT
import ENV
import AC
import torch
import rl_utils
from matplotlib import pyplot as plt
from collections import deque
from TEST import model_test
from RANDOMAGENT import RANDOMAGENT
from copy import deepcopy

np.random.seed(1)
torch.manual_seed(0)

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device=torch.device('cpu')

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=4
num_units=1
bs_cover=1000
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(100,200),
        'sr':random_uniform_int(100,200),
        'tp':random_uniform_float(1,3),
        'pfr':random_uniform_int(900,1000),
        'pf':random_uniform_int(400,500),
        'plr':unit_loc(0,bs_cover),
        'pl':random_loc(0,bs_cover//10),
        'pd':random_uniform_float(2,6),
        'ps':random_uniform_float(60,100),
        'pbr':random_uniform_int(4000,5000),
        'pb':random_uniform_int(2000,4000),
        'ppr':random_uniform_int(200,300),
        'pp':random_uniform_int(100,200),
        'pg':random_uniform_int(20,40),
        'pcr':random_uniform_float(50,300),
        'pc':random_uniform_float(10,100),
        'whitenoise':1,
        'alpha':2}

config_in=deepcopy(config)
config_in['tp']=random_uniform_float(8,10)
num_subtasks=2
time_base=100

deque_1=deque()
proin=ENV_AGENT.PROIN(config_in,1000,deque_1)
proin.run()

deque_2=deque()

env_agent=ENV_AGENT.ENV_AGENT(time_base=time_base,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)

w=(env_agent.num_processors,env_agent.num_processor_attributes-1+env_agent.num_subtasks)
agent=AC.ActorCritic(w,num_subtasks,actor_lr,critic_lr,gamma,device,clip_grad=0.1,conv=1)

env_agent.set_agent(agent)

bigenv=ENV_AGENT.BIGENV_ONE([deque_1],[deque_2],env_agent)
return_list=rl_utils.train_on_policy_agent(bigenv,bigenv.agent,num_episodes)

model_test(bigenv,bigenv.agent,1,num_subtasks,cycles=1)
print('next_agent##################################################')
r_agent=RANDOMAGENT(w,num_subtasks)
model_test(bigenv,r_agent,1,num_subtasks,cycles=1)

'''env=ENV.ENVONE(time_base,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
state=env.reset()
print('state:{}'.format(state[0]))
action=agent.take_action(state)
print('action:{}'.format(action))
next_state, reward, done, _ = env.step(action)
print('next_state:{}'.format(next_state[0]))'''