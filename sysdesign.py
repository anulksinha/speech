import numpy as np
import wave
import random
import json
import pickle
from datetime import datetime, timedelta
from scipy.io import wavfile
from scipy import signal
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

#################################################################################################
## Class Definition
## SystemDesign : Designs the system based on given parameters
##                Currently only handles whole system design
##                Layer based designs to be added in the future
##                All neurons would only have information about incoming connections
#################################################################################################
## Features :
##           Primal Emotion System : 
##                  Calculates the primal emotion of the input for quick actions
##                  Changes the structure and connections of the system
##                  Acts as reinforcement for the response system
##                  Contains:
##                           Positive
##                           Negative
##                           Neutral
##                           Random
##                           Fear
##                           Ignore (Counter-balance for fear)
##
##           Response System       : 
##                  Calculates the response from the input
##                  Provides feedback to primal emotion system
##                  In future implementation, will have detailed emotion system
#################################################################################################
## Input:
##       gen               : Generation Number, used for Genetic Algorithm
##       mutation          : Mutation rate, used for Genetic Algorithm
##       emote_layers      : Number of layers in primal emotion system
##       num_emote_neurons : Number of neurons in primal emotion system
##       resp_layers       : Number of layers in response system
##       num_resp_neurons  : Number of neurons per layer in response system
##       f_max_connections : Forward connection maximum limit
##       b_max_connections : Backward connection maximum limit
##       active_threshold  : Minimum threshold for neuron's input to be active
##       silent_threshold  : Weight threshold below which neuron becomes silent
##       LTDLTP            : Long term Potentiation/Depression threshold
##       max_weight        : Maximum weight of a neuron
##       epsilon           : Value of epsilon for exploration
##       neural_decay      : Rate at which the value in a neuron decays
##       synaptic_diffuse  : Rate at which neurotransmitter is diffused in synaptic space
#################################################################################################
## Output:
##        emote_neural_structure : Primal Emotion Neural structure
##        resp_neural_structure  : Response system neural structure
##
## Output Features:
##        Structure        : Layers,Neuron in that layer, forward/backward,
##                           neuron in forward/backward layer, (0-weight,
##                           1-synaptic space, 2-structure flag)
##                           
##        neuron          : Layers, Neuron in that layer,(0-value,
##                          1-silent threshold, 2-active threshold, 3-LTDLTP,
##                          4-excite probability, 5-inhibit probability,
##                          6-fired/not fired, 7-previous value if not fired )
#################################################################################################

class SystemDesign(object):
    __id_count = 0
    def __init__(self,gen= None,mutation = 0.1,input_dim = 5,emote_layers=3, num_emote_neurons=6,\
                 resp_layers=3, num_resp_neurons=8,f_max_connections=8, b_max_connections=8,\
                 active_threshold=0.00001,silent_threshold=0.001,LTDLTP = 0.3, max_weight=2,\
                 epsilon = 0.1, neural_decay = 0.9, synaptic_diffuse = 0.5, lr = 0.00001,w_up = True):
        
        ## Initialization of parameters
        SystemDesign.__id_count += 1
        self.id = SystemDesign.__id_count
        self.gen = gen
        self.mutation = mutation
        self.input_dim = input_dim
        self.emote_layers = emote_layers
        self.num_emote_neurons = max(num_emote_neurons,input_dim+1)
        self.resp_layers = resp_layers
        self.num_resp_neurons = max(num_resp_neurons,self.num_emote_neurons+self.input_dim)
        self.f_max_connections = f_max_connections
        self.b_max_connections = b_max_connections
        self.active_threshold = active_threshold
        self.silent_threshold = silent_threshold
        self.LTDLTP = LTDLTP
        self.max_weight = max_weight
        self.epsilon = epsilon
        self.neural_decay = neural_decay
        self.synaptic_diffuse = synaptic_diffuse
        self.lr = lr
        self.score = np.zeros(shape=(self.input_dim))
        self.cu_score = 0
        self.output = np.zeros(shape=(self.input_dim))
        self.input = np.zeros(shape=(self.input_dim))
        self.w_up = w_up
        #print('Input '+str(self.input.shape))
        #print('Output '+str(self.output.shape))
        self.fear_ig = 0
        ## Initialization of system
        self.emote_neural_structure = np.zeros(shape=(self.emote_layers,self.num_emote_neurons,\
                                                      2,self.num_emote_neurons,3))
        self.emote_neuron = np.zeros(shape=(self.emote_layers,self.num_emote_neurons,8))
        self.emote_fire = np.zeros(shape=(self.emote_layers,self.num_emote_neurons))
        self.emote_copy = np.zeros(shape=(self.emote_layers,self.num_emote_neurons,\
                                                      2,self.num_emote_neurons))
        self.resp_neural_structure = np.zeros(shape=(self.resp_layers,self.num_resp_neurons,\
                                                2,self.num_resp_neurons,3))
        
        self.response_neuron = np.zeros(shape=(self.resp_layers,self.num_resp_neurons,8))
        self.response_fire = np.zeros(shape=(self.resp_layers,self.num_resp_neurons))
        self.resp_copy = np.zeros(shape=(self.resp_layers,self.num_resp_neurons,\
                                                2,self.num_resp_neurons))
        
        if self.w_up:
            self.w = np.random.uniform(-2,2,size=(4,self.num_emote_neurons+2))
        else:
            self.w = np.zeros(shape=(4,self.num_emote_neurons+2))+1
        
        self.emote_neural_structure.astype(float)
        self.emote_neuron.astype(float)
        self.resp_neural_structure.astype(float)
        self.response_neuron.astype(float)
        self.emote_copy.astype(float)
        self.resp_copy.astype(float)
               

    def structural_connection(self):

        ## Building structural connections for Primal Emotion System
        for i in range(self.emote_layers):

            for j in range(self.num_emote_neurons):
                
                ## Setting neuron parameters
                self.emote_neuron[i,j,1] = self.silent_threshold
                self.emote_neuron[i,j,2] = self.active_threshold
                self.emote_neuron[i,j,3] = self.LTDLTP
                self.emote_neuron[i,j,4] = random.uniform(0,1)
                self.emote_neuron[i,j,5] = random.uniform(0,1)
                
                
                if i!=0:
                    ## Forward Connection from immediate neuron in previous layer
                    self.emote_neural_structure[i,j,0,j,0] = random.uniform(0.5,self.max_weight-0.5)
                    ## Setting the default structure flag
                    self.emote_neural_structure[i,j,0,j,2] = 1
                    
                    if j<self.num_emote_neurons-1 :
                        if i<self.emote_layers-1:
                            ## Backward Connection from adjacent neuron in next layer
                            self.emote_neural_structure[i,j,1,j+1,0] = random.uniform(0.5,self.max_weight-0.5)
                            ## Setting the default structure flag
                            self.emote_neural_structure[i,j,1,j+1,2] = 1
                        
                        ## Forward Connection from right adjacent neuron in previous layer
                        self.emote_neural_structure[i,j,0,j+1,0] = random.uniform(0.5,self.max_weight-0.5)
                        ## Setting the default structure flag
                        self.emote_neural_structure[i,j,0,j+1,2] = 1
                else:
                    if j<self.num_emote_neurons-1 :
                        ## Backward Connection from adjacent neuron in next layer
                        self.emote_neural_structure[i,j,1,j+1,0] = random.uniform(0.5,self.max_weight-0.5)
                        ## Setting the default structure flag
                        self.emote_neural_structure[i,j,1,j+1,2] = 1
                    ## Forward connection of the first layer reserved for input
                    #print(self.input_dim)
                    #print(self.emote_neural_structure[i,j,0,0:50,0])
                    #print(np.random.uniform(0.5,self.max_weight-0.5,size=(self.input_dim)))
                    self.emote_neural_structure[i,j,0,0:self.input_dim,0] = np.random.uniform(0.5,self.max_weight-0.5,size=(self.input_dim))
                    ## Forward connection of the first layer reserved for input 2
                    #self.emote_neural_structure[i][j][0][1][0] = random.uniform(0.5,1.5)
                    ## Forward connection of the first Layer reserved for previous output
                    self.emote_neural_structure[i,j,0,-1,0] = random.uniform(0.5,self.max_weight-0.5)
                    ## Setting the default structure flag
                    self.emote_neural_structure[i,j,0,0:self.input_dim+1,2] = 1
                    #self.emote_neural_structure[i][j][0][1][2] = 1
                    #self.emote_neural_structure[i][j][0][2][2] = 1
        
        ## Building structural connections for Response System
        for ii in range(self.resp_layers):

            for jj in range(self.num_resp_neurons):
                
                ## Setting neuron parameters
                self.response_neuron[ii,jj,1] = self.silent_threshold
                self.response_neuron[ii,jj,2] = self.active_threshold
                self.response_neuron[ii,jj,3] = self.LTDLTP
                self.response_neuron[ii,jj,4] = 0.5
                self.response_neuron[ii,jj,5] = 0.5

                if ii!=0:
                    ## Forward Connection from immediate neuron in previous layer
                    self.resp_neural_structure[ii,jj,0,jj,0] = random.uniform(0.5,self.max_weight-0.5)
                    ## Setting the default structure flag
                    self.resp_neural_structure[ii,jj,0,jj,2] = 1
                    
                    if jj<self.num_resp_neurons-1:
                        if ii<self.resp_layers-1:
                            ## Backward Connection from adjacent neuron in next layer
                            self.resp_neural_structure[ii,jj,1,jj+1,0] = random.uniform(0.5,self.max_weight-0.5)
                            ## Setting the default structure flag
                            self.resp_neural_structure[ii,jj,1,jj+1,2] = 1
                        
                        ## Forward Connection from right adjacent neuron in previous layer
                        self.resp_neural_structure[ii,jj,0,jj+1,0] = random.uniform(0.5,self.max_weight-0.5)
                        ## Setting the default structure flag
                        self.resp_neural_structure[ii,jj,0,jj+1,2] = 1
                    #else:
                    #    if jj%2 == 0:
                            ## Implement output backpath
                    #        pass
                else:
                    if jj<self.num_resp_neurons-1:
                        ## Backward Connection from adjacent neuron in next layer
                        self.resp_neural_structure[ii,jj,1,jj+1,0] = random.uniform(0.5,self.max_weight-0.5)
                        ## Setting the default structure flag
                        self.resp_neural_structure[ii,jj,1,jj+1,2] = 1
                    ## First and second position in first layer reserved for input 
                    
                    self.resp_neural_structure[ii,jj,0,0:self.input_dim,0] = np.random.uniform(0.5,self.max_weight-0.5,size=(self.input_dim))
                    #self.resp_neural_structure[ii,jj,0,1,0] = random.uniform(0.5,self.max_weight-0.5)
                    ## Setting the default structure flag
                    self.resp_neural_structure[ii,jj,0,0:self.input_dim,2] = 1
                    #self.resp_neural_structure[ii,jj,0,1,2] = 1
                    for kk in range(self.num_emote_neurons):
                        ## Output of Primal Emotion system connected to response system
                        self.resp_neural_structure[ii,jj,0,kk+self.input_dim,0] = random.uniform(0.5,self.max_weight-0.5)
                        ## Setting the default structure flag
                        self.resp_neural_structure[ii,jj,0,kk+self.input_dim,2] = 1
                        
    def random_connection(self):
        
        ## Building random connections for Primal Emotion System
        for i in range(self.emote_layers):

            for j in range(self.num_emote_neurons):
                ## Forward path
                for k in range(self.num_emote_neurons):
                    if i<self.emote_layers-1:
                        if self.emote_neural_structure[i,k,1,j,0] >= 2*self.silent_threshold:
                            self.emote_neural_structure[i+1,j,0,k,0] = 0
                        else:
                            if random.randint(0,self.emote_layers+1-i)>0:
                                self.emote_neural_structure[i+1,j,0,k,0] = random.uniform(0.5,1.5)
                ## Backward path
                for k in range(self.num_emote_neurons):
                    if i<self.emote_layers-1:
                        if self.emote_neural_structure[i+1,k,0,j,0] >= 2*self.silent_threshold:
                            self.emote_neural_structure[i,j,1,k,0] = 0
                        else:
                            if random.randint(0,self.emote_layers+1-i)>0:
                                self.emote_neural_structure[i,j,1,k,0] = random.uniform(0.5,1.5)

        ## Building random connections for Response System                        
        for ii in range(self.resp_layers):

            for jj in range(self.num_resp_neurons):
                ## Forward path
                for kk in range(self.num_resp_neurons):
                    if ii<self.resp_layers-1:
                        if self.resp_neural_structure[ii,kk,1,jj,0] >= 2*self.silent_threshold:
                            self.resp_neural_structure[ii+1,jj,0,kk,0] = 0
                        else:
                            if random.randint(0,self.resp_layers+1-ii)>0:
                                self.resp_neural_structure[ii+1,jj,0,kk,0] = random.uniform(0.5,1.5)
                ## Backward path
                for kk in range(self.num_resp_neurons):
                    if ii<self.resp_layers-1:
                        if self.resp_neural_structure[ii+1,kk,0,jj,0] >= 2*self.silent_threshold:
                            self.resp_neural_structure[ii,jj,1,kk,0] = 0
                        else:
                            if random.randint(0,self.resp_layers+1-i)>0:
                                self.resp_neural_structure[ii,jj,1,kk,0] = random.uniform(0.5,1.5)
                                
    def prim_emo_fire(self,input_val):
        #print('emo')
        self.input = np.append(self.input,input_val)
        self.emote_neuron[:,:,0] += self.neural_decay*self.emote_neuron[:,:,7]
        self.emote_neuron[:,:,7] = 0
        self.old_em = self.emote_copy
        #print('Input '+str(self.input.shape))
        #print('Output '+str(self.output.shape))
        for i in range(self.emote_layers):
            for j in range(self.num_emote_neurons):
                if self.emote_neuron[i,j,0] == 0:
                    self.emote_neuron[i,j,6] = 0
                else:
                    self.emote_neuron[i,j,6] = 1
                    if abs(self.emote_neuron[i,j,0]) < self.emote_neuron[i,j,1]:
                        self.emote_neuron[i,j,7] = self.emote_neuron[i,j,0]
                        self.emote_neuron[i,j,0] = 0
        
        self.emote_fire = np.dstack((self.emote_fire,self.emote_neuron[:,:,6]))
        #print('Input '+str(self.input.shape))
        #print('Output '+str(self.output.shape))
        for i in range(self.emote_layers-1,-1,-1):
            for j in range(self.num_emote_neurons-1,-1,-1):
                if i == 0:
                    #print(self.emote_neural_structure[i,j,0,0:self.input_dim,0])
                    #print(self.input_dim)
                    #print(self.input[:])
                    neuro_fsum = np.nan_to_num(np.sum(np.tanh(np.multiply(self.emote_neural_structure[i,j,0,0:self.input_dim,0],self.input[-1:-self.input_dim-1:-1]))))
                    #neuro_fsum += np.nan_to_num(np.sum(np.tanh((self.emote_neural_structure[i,j,0,1,0]*self.input[-2]))))
                    neuro_fsum += np.nan_to_num(np.sum(np.tanh((self.emote_neural_structure[i,j,0,-1,0]*self.output[-1]))))
                    #print(neuro_fsum)
                    
                    #print(self.emote_neural_structure[i,j,0,0,0]*input_val)
                    self.emote_copy[i,j,0,0:self.input_dim] = np.nan_to_num(np.multiply(self.emote_neural_structure[i,j,0,0:self.input_dim,0],self.input[-1:-self.input_dim-1:-1]))
                    #self.emote_copy[i,j,0,1] = np.nan_to_num(np.multiply(self.emote_neural_structure[i,j,0,1,0],self.input[-2]))
                    self.emote_copy[i,j,0,-1] = np.nan_to_num(np.multiply(self.emote_neural_structure[i,j,0,-1,0],self.output[-1]))
                else:
                    neuro_fsum = np.nan_to_num(np.sum(np.tanh(np.multiply\
                                                (self.emote_neural_structure[i,j,0,:,0],self.emote_neuron[i-1,:,0]))))
                    #x=np.multiply(self.emote_neural_structure[i,j,0,:,0],\
                    #                                         self.emote_neuron[i-1,:,0])
                    #print(x)
                    self.emote_copy[i,j,0,:] = np.nan_to_num(np.multiply(self.emote_neural_structure[i,j,0,:,0],\
                                                             self.emote_neuron[i-1,:,0]))
                    
                if i < self.emote_layers-1:
                    neuro_bsum = np.nan_to_num(np.sum(np.tanh(np.multiply\
                                                (self.emote_neural_structure[i,j,1,:,0],self.emote_neuron[i+1,:,0]))))
                    self.emote_copy[i,j,1,:] = np.nan_to_num(np.multiply(self.emote_neural_structure[i,j,1,:,0],\
                                                             self.emote_neuron[i+1,:,0]))
                else:
                    neuro_bsum = 0

                if neuro_fsum + neuro_bsum <= 0:
                    self.emote_neuron[i,j,0] = 0
                else:
                    #print(i,j)
                    self.emote_neuron[i,j,0] = neuro_fsum + neuro_bsum
                    
                    #print(self.emote_neuron[i,j,4],self.emote_neuron[i,j,5])
                    ex_in = random.choices([0,1],[self.emote_neuron[i,j,4],self.emote_neuron[i,j,5]])
                    #print(ex_in)
                    if ex_in[0] == 1:
                        self.emote_neuron[i,j,0] *= -1
                        if self.score[-1]<0:
                            self.emote_neuron[i,j,4] *= 0.5
                            self.emote_neuron[i,j,4] += 0.5
                        else:
                            self.emote_neuron[i,j,5] *= 0.5
                            self.emote_neuron[i,j,5] += 0.5
                    else:
                        if self.score[-1]<0:
                            self.emote_neuron[i,j,5] *= 0.5
                            self.emote_neuron[i,j,5] += 0.5
                        else:
                            self.emote_neuron[i,j,4] *= 0.5
                            self.emote_neuron[i,j,4] += 0.5
                        #print('neg')
                       
            #print(self.emote_copy)
        #print(self.emote_neuron[:,:,0])                
        ## Implement Synaptic space
        self.control_system()        
    def resp_sys_fire(self,input_val):
        #print('rs')
        #neuro_fsum = 0
        
        self.output = np.append(self.output,np.nan_to_num(np.tanh(np.sum(np.nan_to_num(np.tanh(self.response_neuron[-1,:,0]))))))
        neuro_bsum = 0
        self.response_neuron[:,:,0] += self.neural_decay*self.response_neuron[:,:,7]
        self.response_neuron[:,:,7] = 0
        self.old_rs = self.resp_copy
        #print('Input '+str(self.input.shape))
        #print('Output '+str(self.output.shape))
        for i in range(self.resp_layers):
            for j in range(self.num_resp_neurons):
                if self.response_neuron[i,j,0] == 0:
                    self.response_neuron[i,j,6] = 0
                else:
                    self.response_neuron[i,j,6] = 1
                    if abs(self.response_neuron[i,j,0]) < self.response_neuron[i,j,2]:
                        self.response_neuron[i,j,7] = self.response_neuron[i,j,0]
                        self.response_neuron[i,j,0] = 0
        
        self.response_fire = np.dstack((self.response_fire,self.response_neuron[:,:,6]))
        #print('Input '+str(self.input.shape))
        #print('Output '+str(self.output.shape))            
        for i in range(self.resp_layers-1,-1,-1):
            for j in range(self.num_resp_neurons-1,-1,-1):
                if i == 0:
                    neuro_fsum = np.nan_to_num(np.sum(np.tanh(np.multiply(self.resp_neural_structure[i,j,0,0:self.input_dim,0],self.input[-1:-self.input_dim-1:-1]))))
                    
                    #neuro_fsum += np.nan_to_num(np.tanh(self.resp_neural_structure[i,j,0,1,0]*self.input[-2]))
                    self.resp_copy[i,j,0,0:self.input_dim] = np.nan_to_num(np.tanh(np.multiply\
                    (self.resp_neural_structure[i,j,0,0:self.input_dim,0],self.input[-1:-self.input_dim-1:-1])))
                    #self.resp_copy[i,j,0,1] = np.nan_to_num(np.tanh(self.resp_neural_structure[i,j,0,1,0]*self.input[-2]))
                    neuro_fsum += np.nan_to_num(np.sum(np.tanh
                                         (np.multiply(self.resp_neural_structure[i,j,0,self.input_dim:self.num_emote_neurons+self.input_dim,0],self.emote_neuron[-1,:,0]))))
                    self.resp_copy[i,j,0,self.input_dim:self.num_emote_neurons+self.input_dim] = np.nan_to_num(np.multiply\
                    (self.resp_neural_structure[i,j,0,self.input_dim:self.num_emote_neurons+self.input_dim,0],\
                                                              self.emote_neuron[-1,:,0]))
                    
                else:
                    neuro_fsum = np.nan_to_num(np.sum(np.tanh
                                        (np.multiply(self.resp_neural_structure[i,j,0,:,0],self.response_neuron[i-1,:,0]))))
                    self.resp_copy[i,j,0,:] = np.nan_to_num(np.multiply(self.resp_neural_structure[i,j,0,:,0],\
                                                            self.response_neuron[i-1,:,0]))
                    
                if i < self.resp_layers-1:    
                    neuro_bsum = np.nan_to_num(np.sum(np.tanh
                                        (np.multiply(self.resp_neural_structure[i,j,1,:,0],self.response_neuron[i+1,:,0]))))
                    self.resp_copy[i,j,1,:] = np.nan_to_num(np.multiply(self.resp_neural_structure[i,j,1,:,0],\
                                                            self.response_neuron[i+1,:,0]))
                
                if neuro_fsum + neuro_bsum <= 0:
                    self.response_neuron[i,j,0] = 0
                else:
                    self.response_neuron[i,j,0] = neuro_fsum + neuro_bsum
                    
                    ex_in = random.choices([0,1],[self.response_neuron[i,j,4],self.response_neuron[i,j,5]])
                    #print(ex_in.shape)
                    if ex_in[0] == 1:
                        self.response_neuron[i,j,0] *= -1
                        if self.score[-1]<0:
                            self.response_neuron[i,j,4] *= 0.5
                            self.response_neuron[i,j,4] += 0.5
                        else:
                            self.response_neuron[i,j,5] *= 0.5
                            self.response_neuron[i,j,5] += 0.5
                    else:
                        if self.score[-1]<0:
                            self.response_neuron[i,j,5] *= 0.5
                            self.response_neuron[i,j,5] += 0.5
                        else:
                            self.response_neuron[i,j,4] *= 0.5
                            self.response_neuron[i,j,4] += 0.5
                            
                        #print('neg')
                       
        
        
        ## Score update
        tempscore = self.output[-2]-self.input[-1]
        if tempscore < 0.00001:
            tempscore = 0.00001
        self.score = np.append(self.score,tempscore)
        self.cu_score -= np.power(self.score[-1],2)
        ## Implement Synaptic space
        #self.prim_emo_control_system()
        #self.prim_emo_control_system()
    def control_system(self):
        ## To implement LTDLTP, weight update, excitory and inhibitory probability update
        ## Weight Update
        ## Primal Emotion Control
        #print('control')
        #print('Input '+str(self.input.shape))
        #print('Output '+str(self.output.shape))
        if not (self.w_up):
            self.w = np.zeros(shape=(4,self.num_emote_neurons+2))+1
        if len(self.output)>=10+self.emote_layers:
            corrco = np.abs(np.around(np.nan_to_num(np.corrcoef(100000*self.input[-1:-11],100000*self.output[-1:-11]))[0,1], decimals =2))
            delta1 = self.w[0,0]*(1-corrco) + np.sum(np.multiply(self.w[0,1:self.num_emote_neurons+1],self.emote_neuron[-1,:,0]))
            delta2 = self.lr*delta1*self.w[0,self.num_emote_neurons+1]* self.old_em
            if np.isnan(delta1):
                delta2 = -0.00001
            self.emote_neural_structure[:,:,:,:,0] = np.nan_to_num(np.absolute(np.multiply(self.emote_neural_structure[:,:,:,:,0],1+np.tanh(delta2))))
            self.emote_neural_structure[:,:,:,:,0] += 0.001
            
            ## Synaptic scaling           
            for i in range(self.emote_layers):

                for j in range(self.num_emote_neurons):
                    if np.max(self.emote_neural_structure[i,j,:,:,0]) > self.max_weight:
                        #print('max exceeded')
                        self.emote_neural_structure[i,j,:,:,0] /= np.max(self.emote_neural_structure[i,j,:,:,0])
                        self.emote_neural_structure[i,j,:,:,0] = np.nan_to_num(self.emote_neural_structure[i,j,:,:,0]) + 0.001
            
            delta3 = self.w[1,0]*(1-corrco) + np.sum(np.multiply( self.w[1,1:self.num_emote_neurons+1],self.emote_neuron[-1,:,0]))
            if np.isnan(delta3):
                delta3 = -0.00001
            ## Fear or Ignore
            
            #self.emote_neuron[:,:,2] *= 1 + np.tanh(delta3)
            #self.emote_neuron[:,:,2] = np.nan_to_num(np.tanh(self.emote_neuron[:,:,2])) + 0.01

            self.lr *= 1 - np.tanh(delta3)
            self.lr = np.nan_to_num(np.tanh(self.lr))/10+ 0.00001
            
            self.emote_neural_structure[:,:,:,:,:] = np.around(self.emote_neural_structure[:,:,:,:,:],decimals=3)
                       
    
            ## Response System Control 
            if len(self.output)>10+self.resp_layers:
                delta4 =  self.w[2,0]*(1-corrco) + np.sum(np.multiply( self.w[2,1:self.num_emote_neurons+1],self.emote_neuron[-1,:,0]))
                delta5 = self.lr*delta1* self.w[2,self.num_emote_neurons+1]* self.old_rs
                if np.isnan(delta4):
                    delta5 = -0.00001
                self.resp_neural_structure[:,:,:,:,0] = np.nan_to_num(np.absolute(np.multiply(self.resp_neural_structure[:,:,:,:,0],1+np.tanh(delta5))))+ 0.001
                #self.resp_neural_structure[:,:,:,:,0] += 0.001

                ## Synaptic scaling           
                for i in range(self.resp_layers):

                    for j in range(self.num_resp_neurons):
                        if np.max(self.resp_neural_structure[i,j,:,:,0]) > self.max_weight:
                            self.resp_neural_structure[i,j,:,:,0] /= np.max(self.resp_neural_structure[i,j,:,:,0])
                            self.resp_neural_structure[i,j,:,:,0] = np.nan_to_num(self.resp_neural_structure[i,j,:,:,0]) + 0.01
                #self.response_neuron[:,:,2] *= 1 + np.tanh(delta3)
                #self.response_neuron[:,:,2] = np.nan_to_num(np.tanh(self.response_neuron[:,:,2])) + 0.001

                self.resp_neural_structure[:,:,:,:,:] = np.around(self.resp_neural_structure[:,:,:,:,:],decimals=3)

            ## Exploration Control
            if np.random.rand() <= self.epsilon:
                self.random_connection()

            delta6 =  self.w[3,0]*(1-corrco) + np.sum(np.multiply( self.w[3,1:self.num_emote_neurons+1],self.emote_neuron[-1,:,0]))

            if np.isnan(delta6):
                delta6 = -0.00001

            self.epsilon *= 1 + np.tanh(delta3)
            self.epsilon = np.nan_to_num(np.tanh(self.epsilon))+ 0.001

    def w_sgd(self):
        if self.score[-1] <= 0.00001:
            w_d = 0
        else:
            w_d = self.lr*self.score[-1]
        
        self.w -= w_d*self.input[-1]
            
        
    def save_model(self,filename=None):
        if filename == None:
            filename = './results/models/'+str(self.gen)+'_'+str(self.id)+'_'+str(datetime.now()).replace(' ','_').replace(':','_')+'.txt'
        savefile = open(filename,'wb')
        pickle.dump(self,savefile)
        print('\nModel Saved\n')
    def load_model(f_name):
        loadfile = open(f_name,'rb')
        return(pickle.load(loadfile))
            
       
         
    def reset_id(self):
        SystemDesign.__id_count = 0
    
    def system_summary(self):

        print('\nid                                  : '+str(self.id))
        print('Generation                          : '+str(self.gen))
        print('Mutation Rate                       : '+str(self.mutation))
        print('Epsilon                             : '+str(self.epsilon))
        print('Learning Rate                       : '+str(self.lr))
        print('Weight Update                       : '+str(self.w))
        print('Primal Emotion layers               : '+str(self.emote_layers))
        print('Primal Emotion neurons per layer    : '+str(self.num_emote_neurons))
        print('Response system layers              : '+str(self.resp_layers))
        print('Response system neurons per layer   : '+str(self.num_resp_neurons))
        print('MAX Forward connections per neuron  : '+str(self.f_max_connections))
        print('MAX Backward connections per neuron : '+str(self.b_max_connections))
        print('MAX neural connection weight        : '+str(self.max_weight))
        print('Error                               : '+str(self.score))
        print('\nPrimal Emotion System Connection  : \n'+str(self.emote_neural_structure[:,:,:,:,0]))
        print('\nPrimal Emotion System Neuron      : \n'+str(self.emote_neuron[:,:,0]))
        print('\nResponse System Connection        : \n'+str(self.resp_neural_structure[:,:,:,:,0]))
        print('\nResponse System Neuron            : \n'+str(self.response_neuron[:,:,0]))
        
        print('\nPrimal Emotion Neuron threshold   : \n'+str(self.emote_neuron[:,:,2]))
        print('\nResponse Neuron threshold         : \n'+str(self.response_neuron[:,:,2]))
        print('\nPrimal Emotion Neuron excite prob : \n'+str(self.emote_neuron[:,:,4]))
        print('\nResponse Neuron excite prob       : \n'+str(self.response_neuron[:,:,4]))
        print('\nPrimal Emotion Neuron inhibit prob: \n'+str(self.emote_neuron[:,:,5]))
        print('\nResponse Neuron inhibit prob      : \n'+str(self.response_neuron[:,:,5]))
        
'''    
if __name__ == '__main__':

    d1 = datetime.now()
    print('Time started : '+str(d1))
    samplerate, data = wavfile.read('D:/sample_1.wav')
    times = np.arange(len(data))/float(samplerate)
    x = np.max(np.absolute(data))
    print(samplerate)
    print(x)
    print(data.shape)
    #print(data)
    #data_norm = normalize(data.reshape((-1,len(data)))).flatten()
    data_resam = np.multiply(100/x,data)
    print(data_resam.flatten().shape)
    #plt.plot(times,data_resam)
    #plt.show

    #print('resampling')
    #data_resam = signal.resample(data_norm,10000)
    #print('######################################')

    sam_pro = 1000
    Total_gen = 4000
    objlist = []
    list_sc = []
    best_score_gen = []
    logfile = open('./log.txt','w+')
    logfile.write('Time started : '+str(d1))
    for g in range(Total_gen):
        score = []
        c = random.randint(0,len(data_resam)//sam_pro - 1)
        for s in range(10):

            new_net = SystemDesign(gen=g+1)
            objlist.append(new_net)

            ## Select top 2
            if g>0:
                #print('ok g')
                if s == 0:
                    print(list_sc[-1])
                list_sc[-1].sort(key= lambda x:x[1])
                #print(list_sc[-1])
                #print(scorelist)
                for obj in objlist:
                    #print(obj.id)
                    #print(list_sc[-1][-1][0])
                    if obj.gen == g:
                        #print(obj.id)
                        if obj.id == list_sc[-1][-1][0]:
                            #print(obj.id)
                            model1em = obj.emote_neural_structure
                            model1rs = obj.resp_neural_structure
                        elif obj.id == list_sc[-1][-2][0]:
                            #print(obj.id)
                            model2em = obj.emote_neural_structure
                            model2rs = obj.resp_neural_structure


            new_net.structural_connection()
            new_net.random_connection()

            ## Cross Breed
            if s %2 == 0 and g>0:
                #print('ok s0')
                best_score_gen.append([g-1,list_sc[-1][-1][1]])
                new_net.emote_neural_structure[0,:,:,:,0] = model1em[0,:,:,:,0]
                new_net.emote_neural_structure[1,:,:,:,0] = model2em[1,:,:,:,0]
                new_net.emote_neural_structure[2,:,:,:,0] = model1em[2,:,:,:,0]

                new_net.resp_neural_structure[0,:,:,:,0] = model1rs[0,:,:,:,0]
                new_net.resp_neural_structure[1,:,:,:,0] = model2rs[1,:,:,:,0]
                new_net.resp_neural_structure[2,:,:,:,0] = model1rs[2,:,:,:,0]

                for i in range(new_net.emote_layers):
                    for j in range(new_net.num_emote_neurons):
                        if np.random.rand() <= new_net.mutation/10:
                            x=random.randint(0,1)
                            new_net.resp_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)

            if s %2 == 1 and g>0:
                #print('ok s1')
                new_net.emote_neural_structure[0,:,:,:,0] = model2em[0,:,:,:,0]
                new_net.emote_neural_structure[1,:,:,:,0] = model1em[1,:,:,:,0]
                new_net.emote_neural_structure[2,:,:,:,0] = model2em[2,:,:,:,0]

                new_net.resp_neural_structure[0,:,:,:,0] = model2rs[0,:,:,:,0]
                new_net.resp_neural_structure[1,:,:,:,0] = model1rs[1,:,:,:,0]
                new_net.resp_neural_structure[2,:,:,:,0] = model2rs[2,:,:,:,0]

                for i in range(new_net.emote_layers):
                    for j in range(new_net.num_emote_neurons):
                        if np.random.rand() <= new_net.mutation:
                            x=random.randint(0,1)
                            new_net.resp_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)

            if s %5 == 0 and g>1:
                new_net.structural_connection()
                new_net.random_connection()
            #new_net.system_summary()
            logfile.write('\n##########################################################################################\n')
            logfile.write('Generation                          : '+str(new_net.gen)+'\n')
            logfile.write('id                                  : '+(str(new_net.id))+'\n')

            logfile.write('Mutation Rate                       : '+str(new_net.mutation)+'\n')
            logfile.write('Primal Emotion layers               : '+str(new_net.emote_layers)+'\n')
            logfile.write('Primal Emotion neurons per layer    : '+str(new_net.num_emote_neurons)+'\n')
            logfile.write('Response system layers              : '+str(new_net.resp_layers)+'\n')
            logfile.write('Response system neurons per layer   : '+str(new_net.num_resp_neurons)+'\n')
            logfile.write('MAX Forward connections per neuron  : '+str(new_net.f_max_connections)+'\n')
            logfile.write('MAX Backward connections per neuron : '+str(new_net.b_max_connections)+'\n')
            logfile.write('MAX neural connection weight        : '+str(new_net.max_weight)+'\n')

            logfile.write('\nPrimal Emotion System Connection  : \n'+str(new_net.emote_neural_structure[:,:,:,:,0]))
            logfile.write('\nPrimal Emotion System Neuron      : \n'+str(new_net.emote_neuron[:,:,0]))
            logfile.write('\nResponse System Connection        : \n'+str(new_net.resp_neural_structure[:,:,:,:,0]))
            logfile.write('\nResponse System Neuron            : \n'+str(new_net.response_neuron[:,:,0]))

            for d in range(sam_pro):
                #if d%800 == 0:
                    #print('######################################')
                    #print('Generation %i Id %i'%(new_net.gen,new_net.id))
                    #print('data %f %% processed'%((d+1)*100/len(data_resam)))
                    #print('%i ms data processed'%(d//8))
                new_net.prim_emo_fire(data_resam[c*sam_pro + d])
                new_net.resp_sys_fire(data_resam[d])

            total_error = -np.sum(np.power(new_net.score,2))
            print('######################################')
            print('Generation %i Id %i'%(new_net.gen,new_net.id))
            print('Total Error : '+str(total_error))
            print('######################################')
            logfile.write('\nScore                               : '+str(total_error))
            score.append([new_net.id,total_error])
        print('#######################################################################')
        #print(score)
        list_sc.append(score)
    d2 = datetime.now()
    logfile.write('Time Ended : '+str(d1))
    logfile.write('#######################################################################')
    print('Code ran for : '+str(d2-d1))
    logfile.write('Code ran for : '+str(d2-d1))
    fig1 = plt.figure()
    x=np.array(best_score_gen)
    plt.plot(x[:,0],x[:,1])
    plt.xlabel('Generations')
    plt.ylabel('MSE Error')
    plt.title('Genetic Algorithm Optimization')
    plt.show()
    fig1.savefig('./ErrGen.png')
'''