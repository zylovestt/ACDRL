import numpy as np
import scipy.integrate as si

class ADENVBASE:
    attributes=[
            ('processor_frequency','pf'),
            ('processor_consume','pc'),
            ('relative_location','rl'),
            ('relative_speed','rs'),
            ('relative_distancefromroad','rd'),
            ('source_gain','sg'),
            ('processor_bandwidth','pb'),
            ('processor_power','pp'),
            ('processor_wait','pw'),
            ('processor_lastaway_wait','plw'),
            ('subtask_location','sl'),               #改进！：state三个状态

            ('processor_distancefromroad','pd'),
            ('processor_gain','pg'),
            ('processor_speed','ps'),
            ('processor_location','pl'),
            ('time_pass','tp'),
            ('whitenoise','whitenoise'),
            ('alpha','alpha'),
            ('source','source'),

            ('subtask_cycle','sc'),
            ('subtask_returnsize','sr')]

    num_processor_attributes=11
    num_subtask_attributes=2
    attr_code={attr:code for code,attr in attributes}
    brief_code={code:attr for code,attr in attributes}

    def __init__(self,**kwards):
        self.num_processors=kwards['num_processors']
        self.num_subtasks=kwards['num_subtasks']
        self.num_roadsideunits=kwards['num_roadsideunits']
        self.max_num_cars=self.num_processors-self.num_roadsideunits
        self.basestation_cover=kwards['basestation_cover']
        self.config=kwards['config']
        self.train=True

    def reset_pro(self):
        #np.random.seed(1)
        self.done=0
        self.time=0
        self.base_sequence=0
        
        #处理器属性
        num_cars=self.max_num_cars
        num_units=self.num_roadsideunits
        self.processor_frequency=np.zeros(self.num_processors)
        self.processor_frequency[:self.num_roadsideunits]=self.config['pfr'](num_units)
        self.processor_frequency[self.num_roadsideunits:]=self.config['pf'](num_cars)

        self.processor_consume=np.zeros(self.num_processors)
        self.processor_consume[:self.num_roadsideunits]=self.config['pcr'](num_units)
        self.processor_consume[self.num_roadsideunits:]=self.config['pc'](num_cars)

        self.processor_location=np.zeros(self.num_processors)
        self.processor_location[:self.num_roadsideunits]=self.config['plr'](num_units)
        self.processor_location[self.num_roadsideunits:]=self.config['pl'](num_cars)

        self.processor_distancefromroad=np.zeros(self.num_processors)
        self.processor_distancefromroad[self.num_roadsideunits:]=self.config['pd'](num_cars)

        self.processor_speed=np.zeros(self.num_processors)
        self.processor_speed[self.num_roadsideunits:]=self.config['ps'](1)[0]

        self.processor_wait=np.zeros(self.num_processors)
        self.processor_lastaway_wait=np.zeros(self.num_processors)

        #传输属性
        self.processor_bandwidth=np.zeros(self.num_processors)
        self.processor_bandwidth[:self.num_roadsideunits]=self.config['pbr'](num_units)
        self.processor_bandwidth[self.num_roadsideunits:]=self.config['pb'](num_cars)

        self.processor_power=np.zeros(self.num_processors)
        self.processor_power[:self.num_roadsideunits]=self.config['ppr'](num_units)
        self.processor_power[self.num_roadsideunits:]=self.config['pp'](num_cars)

        b=self.config['pg']((self.num_processors,self.num_processors))
        self.processor_gain=(b+b.T)/2
        self.whitenoise=self.config['whitenoise']
        self.alpha=self.config['alpha']
    
    def set_task(self):
        #设置总任务属性
        self.source=self.config['source'](1)[0]    #注意！

        #设置子任务属性
        self.subtask_cycle=self.config['sc'](self.num_subtasks)
        self.subtask_returnsize=self.config['sr'](self.num_subtasks)
        self.subtask_location=np.zeros((self.num_processors,self.num_subtasks),dtype='int')
        #self.subtask_location[:]=1000*np.max(self.subtask_cycle)
        for j in range(self.num_subtasks):                     #注意！
            num_choice_units=np.random.randint(low=0,high=self.num_roadsideunits+1)
            units_choice=np.random.choice(np.arange(self.num_roadsideunits),num_choice_units,replace=False)
            num_choice_cars=np.random.randint(low=1,high=self.max_num_cars+1)
            cars_choice=np.random.choice(np.arange(self.num_roadsideunits,self.num_processors),
                num_choice_cars,replace=False)
            processor_choice=np.hstack((units_choice,cars_choice))
            #processor_choice=np.random.choice(np.arange(self.num_processors),num_choice,replace=False)
            #self.subtask_location[processor_choice,j]=self.subtask_cycle[j]
            self.subtask_location[processor_choice,j]=1

    def caculate_relative(self):
        self.relative_location=self.processor_location-self.processor_location[self.source]
        self.relative_distancefromroad=self.processor_distancefromroad-self.processor_distancefromroad[self.source]
        self.relative_speed=self.processor_speed-self.processor_speed[self.source]
        self.source_gain=self.processor_gain[self.source]

    def packet_return(self):
        F=lambda x: tuple(temp.reshape(-1,1).astype('float32') 
            if len(temp.shape)==1 else temp.astype('float32') for temp in x)
        l_p=[]
        for x in self.attributes[:self.num_processor_attributes]:
            l_p.append(eval('self.'+x[0]))
        state_processors=F(l_p)

        G=lambda x: tuple(temp.reshape(1,-1).astype('float32') for temp in x)
        l_t=[]
        for x in self.attributes[-self.num_subtask_attributes:]:
            l_t.append(eval('self.'+x[0]))
        state_subtasks=G(l_t)

        return ((np.concatenate(state_processors,1).reshape(1,1,self.num_processors,-1),
            np.concatenate(state_subtasks,1)))

    def reset(self):
        self.reset_pro()
        self.set_task()
        self.caculate_relative()
        return self.packet_return()

    def stragety(self,action):
        #action是形状为2*num_substasks的数组，第一行表示子任务分配在哪个处理器上，
        #第二个表示子任务的优先级，
        #因为可能多个子任务分配到同一个处理器上

        #任务处理时延
        '''time_execution=self.subtask_location[action[0],range(self.num_subtasks)]\
            /self.processor_frequency[action[0]]'''
        time_execution=self.subtask_cycle/self.processor_frequency[action[0]]

        #任务等待、返回时延
        time_wait=np.zeros(self.num_subtasks)
        time_return=np.zeros(self.num_subtasks)
        time_total=np.zeros(self.num_subtasks)
        eps=1e-4

        if self.done:
            print("The ENV has done")
            raise InterruptedError

        for i in range(self.num_subtasks):
            for j in range(self.num_subtasks):
                if action[1,j]==i:
                    processor=action[0,j]
                    if not self.judge(processor,j):
                        self.done=1
                        print('wrong')
                        time_wait[j]=np.inf
                        return time_execution,time_wait,time_return

                    wait_to_return=max(self.processor_lastaway_wait[processor]-time_execution[j],0)
                    time_wait[j]=self.processor_wait[processor]+wait_to_return

                    relative_speed=self.relative_speed[processor]
                    relative_loction=self.relative_location[processor]\
                        +(time_execution[j]+time_wait[j])*relative_speed    #已改正！

                    distance=lambda x: (relative_loction+x*relative_speed)**2\
                                        +(self.relative_distancefromroad[processor])**2+eps

                    return_rate=lambda x:self.processor_bandwidth[processor]\
                                *np.log2(1+self.processor_power[processor]\
                                    *self.source_gain[processor]
                                    /(self.whitenoise*distance(x)**(self.alpha/2)))
                    base_rate=return_rate(self.basestation_cover/self.processor_speed[self.source]*2)+eps

                    if relative_speed==0:
                        time_return[j]=self.subtask_returnsize[j]/(base_rate-eps)
                    else:
                        time_return[j]=self.findzero(return_rate,
                            self.subtask_returnsize[j]/base_rate+eps,
                            self.subtask_returnsize[j],
                            error_max=1e-4)
                    time_total[j]=time_execution[j]+time_wait[j]+time_return[j]
                    if time_total[j]==np.inf:
                        self.done=1
                        print("time is too long")
                        return time_execution,time_wait,time_return
                    
                    self.processor_wait[processor]+=time_execution[j]
                    self.processor_lastaway_wait[processor]=wait_to_return+time_return[j]
        return time_execution,time_wait,time_return

    def status_change(self):
        #下一个任务
        time_pass=self.config['tp'](1)[0]
        self.time+=time_pass

        #位置改变
        self.processor_location+=self.processor_speed*time_pass

        #范围检测
        loc=time_pass*self.processor_speed+self.processor_location
        if max(loc)>self.basestation_cover:
            #print('out of No.{} BS, get into No.{} BS'.format(self.base_sequence,self.base_sequence+1))
            self.base_sequence+=1
            self.processor_location[self.num_roadsideunits:]-=self.basestation_cover
            #self.processor_location[:self.num_roadsideunits]=self.config['plr'](self.num_roadsideunits)
            if (self.base_sequence)==10:
                self.done=1
                print('success')
        
        #队列等待时间改变
        self.processor_wait-=time_pass
        #dic={pro:[sub for sub in range(self.num_subtasks) if action[0,sub]==pro] for pro in set(action[0])}
        for processor in range(self.num_processors):
            a=self.processor_wait[processor]
            #for sub in dic.setdefault(processor,[]):
                #a+=time_execution[sub]
            if a>=0:
                self.processor_wait[processor]=a
            else:
                self.processor_wait[processor]=0
                b=self.processor_lastaway_wait[processor]+a
                self.processor_lastaway_wait[processor]=b if b>=0 else 0

    def cal_reward(self,time_execution,time_wait,time_return,action,weights):
        #不能删除
        raise NotImplementedError

    def test(self,time_execution,time_wait,time_return):
        raise NotImplemented
    
    def step(self,action):
        time_execution,time_wait,time_return=self.stragety(action)
        self.status_change()
        self.set_task()
        self.caculate_relative()
        if self.train:
            reward=self.cal_reward(time_execution,time_wait,time_return,action,self.weights)
        else:
            reward=self.test(time_execution,time_wait,time_return)
        return self.packet_return(),reward,self.done,None
    
    def judge(self,processor,subtask):
        #processor不能删，子类调用
        return self.subtask_location[processor,subtask]
    
    def findzero(self,fx,x,u,error_max=1e-1):
        F=lambda f:(lambda x:si.quad(f,0,x)[0])
        Fx=lambda x:F(fx)(x)-u
        if Fx(x)<0:
            return np.inf
        l,r=0,x
        mid=(l+r)/2
        y=Fx(mid)
        while abs(y)>error_max:
            if y>0: r=mid
            else: l=mid
            mid=(l+r)/2
            y=Fx(mid)
        return mid

class ENVONE(ADENVBASE):
    def __init__(self,time_base,weights,**kwards):
        super().__init__(**kwards)
        self.time_base=time_base
        self.weights=weights

    def cal_reward(self,time_execution,time_wait,time_return,action,weights):
        a=max(time_execution+time_wait+time_return)
        #print(a)
        r=np.zeros(len(weights))
        total_time=self.time_base*1000 if a==np.inf else a
        r[0]=(self.time_base-total_time)/self.time_base
        r[1]=-np.sum(time_execution*self.processor_consume[action[0]])
        b=np.zeros(self.num_processors)
        for i in range(self.num_subtasks):
            b[action[0][i]]+=time_execution[i]
        r[2]=-np.sum(b)
        r[3]=-np.std(b)
        r[4]=-np.mean(self.processor_wait)
        r[5]=-np.std(self.processor_wait)
        r[6]=-np.mean(self.processor_lastaway_wait)
        r[7]=-np.std(self.processor_lastaway_wait)
        return np.sum(r*weights)
    
    def test(self,time_execution,time_wait,time_return):
        a=max(time_execution+time_wait+time_return)
        total_time=self.time_base*1000 if a==np.inf else a
        return total_time
