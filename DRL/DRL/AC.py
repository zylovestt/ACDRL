from matplotlib.pyplot import cla
import torch
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
import numpy as np
import AGENT_NET

class ActorCritic:
    def __init__(self,input_shape:tuple,num_subtasks,actor_lr,critic_lr,gamma,device,clip_grad,beta,conv):
        if conv:
            self.actor=AGENT_NET.PolicyNet(input_shape,num_subtasks).to(device)
            self.critic=AGENT_NET.ValueNet(input_shape,num_subtasks).to(device)
        else:
            self.actor=AGENT_NET.PolicyNet_FC(input_shape,num_subtasks).to(device)
            self.critic=AGENT_NET.ValueNet_FC(input_shape,num_subtasks).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr,eps=1e-3)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr,eps=1e-3)
        #self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        #self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.beta=beta
        self.clip_grad=clip_grad
        self.device=device
    
    def take_action(self,state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        probs_subtasks_orginal,probs_prior_orginal=self.actor(state)
        '''probs_subtasks_orginal*=[x*y
            for x,y in zip(probs_subtasks_orginal,state[0][0,0,:,-self.num_subtasks:].T)]'''
        action_subtasks=[torch.distributions.Categorical(x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action
    
    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        dones=F(transition_dict['dones']).view(-1,1)

        # 时序差分目标
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)  # 时序差分误差
        probs=self.actor(states)
        s=0
        #probs=(probs[0]+1e-10,probs[1]+1e-10)
        for prob in probs[0]:
            s+=(prob*prob.log()).sum(dim=1)
        t=0
        for prob in probs[1]:
            t+=(prob*prob.log()).sum(dim=1)
        u=self.beta*(s.mean()+t.mean())
        log_probs=torch.log(self.calculate_probs(probs,actions))
        actor_loss=torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss=torch.mean(FU.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        #print(actor_loss)
        #print(critic_loss)
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        nn_utils.clip_grad_norm_(self.actor.parameters(),self.clip_grad)
        nn_utils.clip_grad_norm_(self.critic.parameters(),self.clip_grad)
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
    
    def calculate_probs(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[0][i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        probs=F(0)

        G=lambda i:((torch.gather(out_puts[1][i],1,actions[1][:,[i]])+1e-7)
            /(out_puts[1][i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True)+1e-7)*G(i+1)
            if i<self.num_subtasks else 1.0)
        probs*=G(0)
        return probs

class ACTWOSTEPS(ActorCritic):
    def __init__(self, input_shape: tuple, num_subtasks, actorf_lr, actors_lr, critic_lr, gamma, device, clip_grad, beta):
        self.actorf=AGENT_NET.PolicyNet_First(input_shape,num_subtasks).to(device)
        self.actors=AGENT_NET.PolicyNet_Second(input_shape,num_subtasks).to(device)
        self.critic=AGENT_NET.ValueNet(input_shape,num_subtasks).to(device)
        self.actorf_optimizer=torch.optim.Adam(self.actorf.parameters(),lr=actorf_lr,eps=1e-3)
        self.actors_optimizer=torch.optim.Adam(self.actors.parameters(),lr=actors_lr,eps=1e-3)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr,eps=1e-3)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.beta=beta
        self.clip_grad=clip_grad
        self.device=device

    def take_action(self,state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        probs_subtasks_orginal=self.actorf(state)
        probs_prior_orginal=self.actors(state)
        action_subtasks=[torch.distributions.Categorical(x).sample().item() for x in probs_subtasks_orginal]
        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))
        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        dones=F(transition_dict['dones']).view(-1,1)

        # 时序差分目标
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)  # 时序差分误差
        probs_subtasks_orginal=self.actorf(states)
        probs_prior_orginal=self.actors(states)
        F=lambda i,x:(x[i]*x[i].log()).sum(dim=1)+F(i+1,x) if i<len(x) else 0
        f=self.beta*F(0,probs_subtasks_orginal).mean()
        s=self.beta*F(0,probs_prior_orginal).mean()
        log_probs_f=torch.log(self.calculate_probs_f(probs_subtasks_orginal,actions))
        actor_loss_f=torch.mean(-log_probs_f * td_delta.detach())+f
        log_probs_s=torch.log(self.calculate_probs_s(probs_prior_orginal,actions))
        actor_loss_s=torch.mean(-log_probs_s * td_delta.detach())+s
        # 均方误差损失函数
        critic_loss=torch.mean(FU.mse_loss(self.critic(states), td_target.detach()))
        self.actorf_optimizer.zero_grad()
        self.actors_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        #print(actor_loss)
        #print(critic_loss)
        actor_loss_f.backward()  # 计算策略网络的梯度
        actor_loss_s.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        nn_utils.clip_grad_norm_(self.actorf.parameters(),self.clip_grad)
        nn_utils.clip_grad_norm_(self.actors.parameters(),self.clip_grad)
        nn_utils.clip_grad_norm_(self.critic.parameters(),self.clip_grad)
        self.actorf_optimizer.step()  # 更新策略网络的参数
        self.actors_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
    
    def calculate_probs_f(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        return F(0)
    
    def calculate_probs_s(self,out_puts,actions):
        F=lambda i:(torch.gather(out_puts[i],1,actions[1][:,[i]])
            /(out_puts[i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[i],1,actions[1][:,:i]).sum(axis=1,keepdim=True))*F(i+1)
            if i<self.num_subtasks else 1.0)
        return F(0)
    
class ActorCritic_Double:
    def __init__(self,input_shape:tuple,num_subtasks,lr,weights,gamma,device,clip_grad,beta,conv):
        self.agent=AGENT_NET.DoubleNet(input_shape,num_subtasks).to(device)
        self.agent_optimizer=torch.optim.Adam(self.agent.parameters(),lr=lr,eps=1e-3)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.beta=beta
        self.clip_grad=clip_grad
        self.weights=weights
        self.device=device
    
    def take_action(self,state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        '''probs_subtasks_orginal*=[x*y
            for x,y in zip(probs_subtasks_orginal,state[0][0,0,:,-self.num_subtasks:].T)]'''
        action_subtasks=[torch.distributions.Categorical(x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action
    
    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        dones=F(transition_dict['dones']).view(-1,1)
        # 时序差分目标
        td_target=rewards+self.gamma*self.agent(next_states)[1]*(1-dones)
        td_delta=td_target-self.agent(states)[1]  # 时序差分误差
        probs,_=self.agent(states)
        s=0
        #probs=(probs[0]+1e-10,probs[1]+1e-10)
        for prob in probs[0]:
            s+=(prob*prob.log()).sum(dim=1)
        t=0
        for prob in probs[1]:
            t+=(prob*prob.log()).sum(dim=1)
        epo_loss=self.beta*(s.mean()+t.mean())
        log_probs=torch.log(self.calculate_probs(probs,actions))
        actor_loss=torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss=torch.mean(FU.mse_loss(self.agent(states)[1], td_target.detach()))
        agent_loss=epo_loss+actor_loss+self.weights*critic_loss
        self.agent_optimizer.zero_grad()
        agent_loss.backward()
        nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
        self.agent_optimizer.step()  # 更新策略网络的参数
    
    def calculate_probs(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[0][i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        probs=F(0)

        G=lambda i:((torch.gather(out_puts[1][i],1,actions[1][:,[i]])+1e-7)
            /(out_puts[1][i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True)+1e-7)*G(i+1)
            if i<self.num_subtasks else 1.0)
        probs*=G(0)
        return probs