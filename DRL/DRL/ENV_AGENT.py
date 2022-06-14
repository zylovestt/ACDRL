from collections import deque
import ENV
import numpy as np
import copy

PRO_ATTRIBUTES=[
        ('processor_frequency','pf'),
        ('processor_consume','pc'),
        ('processor_bandwidth','pb'),
        ('processor_power','pp'),
        ('processor_distancefromroad','pd'),
        ('processor_speed','ps'),
        ('processor_location','pl'),
        ('processor_wait','pw'),
        ('processor_lastaway_wait','plw')]

class PROCESSER():
    def __init__(self,dic:dict,timeinqueue,time):
       self.dic=dic
       self.timeinqueue=timeinqueue
       self.time=time 

class PROIN():
    def __init__(self,config:dict,timeover,queueout:deque):
        self.config=config
        self.timeover=timeover
        self.queueout=queueout
        self.time=0

    def run(self):
        while self.time<self.timeover:
            time_pass=self.config['tp'](1)[0]
            self.time+=time_pass
            dic={key:self.config[value](1)[0] for key,value in PRO_ATTRIBUTES[:-2]}
            dic['processor_location']=0
            dic['processor_wait']=0
            dic['processor_lastaway_wait']=0
            proin=PROCESSER(dic,self.time,self.time)
            self.queueout.append(proin)

class PROOUT():
    def __init__(self,queuein:deque):
        self.queuein=queuein        

    def run(self):
        while len(self.queuein):
            self.queuein.popleft()
            print('a car out')

class ENV_AGENT(ENV.ADENVBASE):
    def __init__(self,time_base,**kwards):
        super().__init__(**kwards)
        self.time_base=time_base
        self.agent=None

    def reset(self,queues_in:list[deque[PROCESSER]],queues_out:list[deque[PROCESSER]]):
        self.set_queue(queues_in,queues_out)
        self.reset_pro()
        self.set_task()
        self.caculate_relative()
        return self.packet_return()

    def set_queue(self,queues_in:list[deque[PROCESSER]],queues_out:list[deque[PROCESSER]]):
        self.queues_in=queues_in
        self.queues_out=queues_out

    def set_agent(self,agent):
        self.agent=agent

    def reset_pro(self):
        super().reset_pro()
        #u=slice(self.num_roadsideunits,None)
        self.nopro_index=set(range(self.num_roadsideunits,self.num_processors))
        self.pro_index=set(range(self.num_roadsideunits))
        #self.processor_frequency[list(self.nopro_index)]=1e-4
        #self.processor_speed[list(self.nopro_index)]=0

    def t_change(self,t,processor):
        #改变wait和lastaway_wait
        a=self.processor_wait[processor]-t
        if a>=0:
            self.processor_wait[processor]=a
        else:
            self.processor_wait[processor]=0
            b=self.processor_lastaway_wait[processor]+a
            self.processor_lastaway_wait[processor]=b if b>=0 else 0

    def set_task(self):
        while len(self.pro_index)==self.num_roadsideunits:
            self.time+=1
            flag=0
            for queue in self.queues_in:
                if len(queue)>0:
                    flag=1
                    pro=queue.popleft()
                    while pro.timeinqueue<self.time:
                        index=self.nopro_index.pop()
                        self.pro_index.add(index)
                        t=self.time-pro.time
                        self.copy(index,pro)
                        self.processor_location[index]+=self.processor_speed[index]*t   #注意！
                        self.t_change(t,index)
                        if len(queue)>0:
                            pro=queue.popleft()
                        else:
                            break
                    if pro.timeinqueue>=self.time:
                        queue.appendleft(pro)
            if flag==0:
                self.done=1
                print('game over!')
                return

        #设置总任务属性
        self.source=np.random.choice(np.array(list(self.pro_index-set(list(range(self.num_roadsideunits))))))

        #设置子任务属性
        self.subtask_cycle=self.config['sc'](self.num_subtasks)
        self.subtask_returnsize=self.config['sr'](self.num_subtasks)
        self.subtask_location=np.zeros((self.num_processors,self.num_subtasks),dtype='int')
        self.subtask_location[:]=100*np.max(self.subtask_cycle)
        for j in range(self.num_subtasks):
            num_choice_units=np.random.randint(low=0,high=self.num_roadsideunits+1)
            units_choice=np.random.choice(np.arange(self.num_roadsideunits),num_choice_units,replace=False)
            num_choice_cars=np.random.randint(low=1,high=len(self.pro_index)-self.num_roadsideunits+1)
            cars_choice=np.random.choice(np.array(list(self.pro_index-set(list(range(self.num_roadsideunits))))),
                num_choice_cars,replace=False)
            processor_choice=np.hstack((units_choice,cars_choice))
            #num_choice=np.random.randint(low=1,high=len(self.pro_index)+1)
            #processor_choice=np.random.choice(np.array(list(self.pro_index)),num_choice,replace=False)
            self.subtask_location[processor_choice,j]=self.subtask_cycle[j]

    def judge(self,processor):
        return 1
    
    def status_change(self):  
        for queue in self.queues_in:
            if len(queue)>0:
                pro=queue.popleft()
                while pro.timeinqueue<=self.time:
                    index=self.nopro_index.pop()
                    self.pro_index.add(index)
                    self.copy(index,pro)
                    t=self.time-pro.time
                    self.processor_location[index]+=self.processor_speed[index]*t    #注意！
                    self.t_change(t,index)
                    if len(queue)>0:
                        pro=queue.popleft()
                    else:
                        break
                if pro.timeinqueue>self.time:
                    queue.appendleft(pro)

        time_pass=self.config['tp'](1)[0]
        self.processor_location+=self.processor_speed*time_pass
        temp_set=self.pro_index.copy()
        for pro_index in temp_set:
            if self.processor_location[pro_index]>=self.basestation_cover:
                self.processor_location[pro_index]-=self.basestation_cover
                outpro=PROCESSER(self.to_dict(pro_index),
                    self.time+time_pass-self.processor_location[pro_index]/self.processor_speed[pro_index],
                    self.time)
                No_outqueue=np.random.randint(0,len(self.queues_out))
                self.queues_out[No_outqueue].append(outpro)
                #self.processor_frequency[pro_index]=1e-4
                #self.processor_speed[pro_index]=0
                self.pro_index.remove(pro_index)
                self.nopro_index.add(pro_index)
            else:
                self.t_change(time_pass,pro_index)
        self.time+=time_pass

    def cal_reward(self,time_execution,time_wait,time_return):
        a=max(time_execution+time_wait+time_return)
        total_time=self.time_base*10 if a==np.inf else a
        reward=(self.time_base-total_time)/self.time_base
        return reward

    def copy(self,pro_index,pro):
        self.processor_frequency[pro_index]=pro.dic['processor_frequency']
        self.processor_consume[pro_index]=pro.dic['processor_consume']
        self.processor_bandwidth[pro_index]=pro.dic['processor_bandwidth']
        self.processor_power[pro_index]=pro.dic['processor_power']
        self.processor_wait[pro_index]=pro.dic['processor_wait']
        self.processor_lastaway_wait[pro_index]=pro.dic['processor_lastaway_wait']
        self.processor_distancefromroad[pro_index]=pro.dic['processor_distancefromroad']
        #speed统一
        #self.processor_speed[pro_index]=pro.dic['processor_speed']
        self.processor_location[pro_index]=pro.dic['processor_location']
        self.processor_gain[pro_index]=self.config['pg'](self.num_processors)
    
    def to_dict(self,pro_index):
        dic={}
        for key,_ in PRO_ATTRIBUTES:
            a=eval('self.'+key)
            dic[key]=a[pro_index]
        return dic

class BIGENV_ONE():
    def __init__(self,queues_in:list[deque[PROCESSER]],queues_out:list[deque[PROCESSER]],env_agent:ENV_AGENT):
        self.static_queues_in=queues_in
        self.static_queues_out=queues_out
        self.env_agent=env_agent
        self.agent=env_agent.agent

    def reset(self):
        return self.env_agent.reset(copy.deepcopy(self.static_queues_in),copy.deepcopy(self.static_queues_out))

    def step(self,action):
        return self.env_agent.step(action)
