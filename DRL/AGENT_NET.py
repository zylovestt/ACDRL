import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class PolicyNet_First(nn.Module):
    def __init__(self,input_shape,num_subtasks):
        super(PolicyNet_First,self).__init__()
        self.base_row=nn.Conv2d(1,128,kernel_size=(1,input_shape[1]),stride=1)
        self.base_col=nn.Conv2d(1,128,kernel_size=(input_shape[0],1),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+2*num_subtasks
        self.fc=nn.Sequential(
            nn.ReLU(),
            nn.Linear(conv_out_size,128),
            nn.ReLU(),
            nn.Linear(128,128))
        F=lambda x:nn.Sequential(
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,x))
        self.policy_layer=nn.ModuleList([F(input_shape[0]) for _ in range(num_subtasks)])
    
    def _get_conv_out(self,input_shape):
        o_row=self.base_row(torch.zeros(1,1,*input_shape))
        o_col=self.base_col(torch.zeros(1,1,*input_shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        x=tuple(x[i]/10000 for i in range(len(x)))
        conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
        conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        out_fc=self.fc(conv_out)
        l=[F.softmax(layer(out_fc),dim=1) for layer in self.policy_layer]
        return l

class PolicyNet_Second(nn.Module):
    def __init__(self,input_shape,num_subtasks):
        super(PolicyNet_Second,self).__init__()
        self.base_row=nn.Conv2d(1,128,kernel_size=(1,input_shape[1]),stride=1)
        self.base_col=nn.Conv2d(1,128,kernel_size=(input_shape[0],1),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+2*num_subtasks
        self.fc=nn.Sequential(
            nn.ReLU(),
            nn.Linear(conv_out_size,128),
            nn.ReLU(),
            nn.Linear(128,128))
        F=lambda x:nn.Sequential(
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,x))
        self.prior_layer=nn.ModuleList([F(num_subtasks) for _ in range(num_subtasks)])
    
    def _get_conv_out(self,input_shape):
        o_row=self.base_row(torch.zeros(1,1,*input_shape))
        o_col=self.base_col(torch.zeros(1,1,*input_shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        x=tuple(x[i]/10000 for i in range(len(x)))
        conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
        conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        out_fc=self.fc(conv_out)
        l=[F.softmax(layer(out_fc),dim=1) for layer in self.prior_layer]
        return l


class PolicyNet(nn.Module):
    def __init__(self,input_shape,num_subtasks):
        super(PolicyNet,self).__init__()
        self.num_processors=input_shape[0]
        self.num_attributes=input_shape[1]
        self.num_subtasks=num_subtasks
        hs=200
        self.base_row=nn.Conv2d(1,hs,kernel_size=(1,input_shape[1]),stride=1)
        self.base_col=nn.Conv2d(1,hs,kernel_size=(input_shape[0],1),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+2*num_subtasks
        self.fc=nn.Sequential(
            nn.ReLU(),
            nn.Linear(conv_out_size,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,hs))
        F=lambda x,y:nn.Sequential(
            nn.ReLU(),
            nn.Linear(x,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,y))
        self.policy_layer=nn.ModuleList([F(hs+self.num_processors,input_shape[0]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(hs,num_subtasks) for _ in range(num_subtasks)])
    
    def _get_conv_out(self,input_shape):
        o_row=self.base_row(torch.zeros(1,1,*input_shape))
        o_col=self.base_col(torch.zeros(1,1,*input_shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        #x=tuple(x[i]/10000 for i in range(len(x)))
        conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
        conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        out_fc=self.fc(conv_out)
        l1=[]
        for layer,i in zip(self.policy_layer,range(self.num_subtasks)):
            u=x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)
            v=torch.cat((out_fc,u),dim=1)
            w=layer(v)
            z=F.softmax(w,dim=1)
            p=(1/(-u))-1+w
            z=F.softmax(p,dim=1)+1e-14
            l1.append(z)
        '''l1=[F.softmax(layer(torch.cat((out_fc,x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)),1))
            +(1/(x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)-1e-14)-1),dim=1)
            for layer,i in zip(self.policy_layer,range(self.num_subtasks))]'''
        l2=[F.softmax(layer(out_fc),dim=1) for layer in self.prior_layer]
        return l1,l2
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class ValueNet(nn.Module):      #注意！
    def __init__(self,input_shape,num_subtasks):
        super(ValueNet,self).__init__()
        self.num_processors=input_shape[0]
        self.num_attributes=input_shape[1]
        self.num_subtasks=num_subtasks
        hs=300
        self.base_row=nn.Conv2d(1,hs,kernel_size=(1,input_shape[1]),stride=1)
        self.base_col=nn.Conv2d(1,hs,kernel_size=(input_shape[0],1),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+2*num_subtasks
        self.fc=nn.Sequential(
            nn.ReLU(),
            nn.Linear(conv_out_size,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,hs),
            nn.ReLU(),
            nn.Linear(hs,1))
    
    def _get_conv_out(self,input_shape):
        o_row=self.base_row(torch.zeros(1,1,*input_shape))
        o_col=self.base_col(torch.zeros(1,1,*input_shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        #x=tuple(x[i]/10000 for i in range(len(x)))
        '''u=x[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in x[1]:
            i[:]=(i-i.mean())/i.std()'''
        conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
        conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        return self.fc(conv_out)

class PolicyNet_FC(nn.Module):
    def __init__(self,input_shape,num_subtasks):
        super().__init__()
        input_size=input_shape[0]*input_shape[1]+2*num_subtasks
        self.fc=nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128))
        F=lambda x:nn.Sequential(
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,x))
        self.policy_layer=nn.ModuleList([F(input_shape[0]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(num_subtasks) for _ in range(num_subtasks)])
    
    def _get_conv_out(self,input_shape):
        o_row=self.base_row(torch.zeros(1,1,*input_shape))
        o_col=self.base_col(torch.zeros(1,1,*input_shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        x=tuple(x[i]/x[i].sum() for i in range(len(x)))
        x=torch.cat(tuple(x[i].view(x[i].shape[0],-1) for i in range(len(x))),1)
        out_fc=self.fc(x)
        l1=[F.softmax(layer(out_fc),dim=1) for layer in self.policy_layer]
        l2=[F.softmax(layer(out_fc),dim=1) for layer in self.prior_layer]
        return l1,l2
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

class ValueNet_FC(nn.Module):
    def __init__(self,input_shape,num_subtasks):
        super().__init__()
        input_size=input_shape[0]*input_shape[1]+2*num_subtasks
        self.fc=nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1))
    
    def forward(self,x):
        x=tuple(x[i]/x[i].sum() for i in range(len(x)))
        x=torch.cat(tuple(x[i].view(x[i].shape[0],-1) for i in range(len(x))),1)
        return self.fc(x)