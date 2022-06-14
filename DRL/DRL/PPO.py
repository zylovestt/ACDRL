import torch
import numpy as np
import AGENT_NET
import torch.nn.functional as FU
import numpy as np
import torch.nn.utils as nn_utils
import matplotlib.pyplot as plt
import rl_utils

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self,input_shape,num_subtasks,actor_lr,critic_lr,gamma,device,lmbda,epochs,eps,conv):
        if conv:
            self.actor=AGENT_NET.PolicyNet(input_shape,num_subtasks).to(device)
            self.critic=AGENT_NET.ValueNet(input_shape,num_subtasks).to(device)
        else:
            self.actor=AGENT_NET.PolicyNet_FC(input_shape,num_subtasks).to(device)
            self.critic=AGENT_NET.ValueNet_FC(input_shape,num_subtasks).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

        self.num_subtasks=num_subtasks
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        
        self.device = device

    def take_action(self, state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        probs_subtasks_orginal,probs_prior_orginal=self.actor(state)
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

    def update(self, transition_dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        dones=F(transition_dict['dones']).view(-1,1)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_delta = td_target - self.critic(states)

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.calculate_probs(self.actor(states),actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.calculate_probs(self.actor(states),actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                FU.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            #nn_utils.clip_grad_norm_(self.actor.parameters(),self.clip_grad)
            #nn_utils.clip_grad_norm_(self.critic.parameters(),self.clip_grad)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
    def calculate_probs(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[0][i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        probs=F(0)

        G=lambda i:(torch.gather(out_puts[1][i],1,actions[1][:,[i]])
            /(out_puts[1][i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True))*G(i+1)
            if i<self.num_subtasks else 1.0)
        probs*=G(0)
        return probs