import numpy as np
import ENV
random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=5
num_units=3
bs_cover=1000
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(100,1000),
        'sr':random_uniform_int(10,20),
        'tp':random_uniform_float(0.1,0.2),
        'pfr':random_uniform_int(500,1000),
        'pf':random_uniform_int(200,500),
        'plr':unit_loc(0,bs_cover),
        'pl':random_loc(0,bs_cover//10),
        'pd':random_uniform_float(2,6),
        'ps':random_uniform_float(60,100),
        'pbr':random_uniform_int(200,500),
        'pb':random_uniform_int(200,400),
        'ppr':random_uniform_int(200,300),
        'pp':random_uniform_int(100,200),
        'pg':random_uniform_int(20,40),
        'pcr':random_uniform_float(50,300),
        'pc':random_uniform_float(10,100),
        'whitenoise':1,
        'alpha':2
}
#ENV.ADENVBASE.attr_code
np.random.seed(1)
num_subtasks=7
env=ENV.ADENVBASE(num_cars+num_units,num_subtasks,num_units,bs_cover,config)
env.reset()
for i in range(300):
    l=[np.where(k>0)[0] for k in env.subtask_location.T]
    u=[np.random.choice(x) for x in l]
    action=np.zeros((2,num_subtasks),dtype='int')
    action[0]=u
    a={pro:[sub for sub in range(num_subtasks) if action[0,sub]==pro] for pro in set(action[0])}
    for pro,sub in a.items():
        for i,x in enumerate(sub):
            action[1,x]=i
    _,(time_exe,time_wait,time_return),done=env.step(action)
    print('action: ',action[0])
    print('time_exec: ',['{:<5.5f}'.format(t) for t in time_exe])
    print('time_wait: ',['{:<5.5f}'.format(t) for t in time_wait])
    print('time_return: ',['{:<5.5f}'.format(t) for t in time_return])
    wait=['{:<5.5f}'.format(num) for num in env.processor_wait]
    return_wait=['{:<5.5f}'.format(num) for num in env.processor_lastaway_wait]
    print('processor_wait: ',wait)
    print('return_wait: ',return_wait)
    print(done)
    if done:
        env.reset()
