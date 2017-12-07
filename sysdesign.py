import numpy as np
import wave
import random

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

class SystemDesign:
    __id_count = 0
    def __init__(self,gen= None,mutation = 0.01,emote_layers=3, num_emote_neurons=6,\
                 resp_layers=3, num_resp_neurons=8,f_max_connections=8, b_max_connections=8,\
                 active_threshold=0.1,silent_threshold=0.001,LTDLTP = 0.3, max_weight=2,\
                 epsilon = 0.1, neural_decay = 0.9, synaptic_diffuse = 0.5, lr = 0.01):
        
        ## Initialization of parameters
        SystemDesign.__id_count += 1
        self.id = SystemDesign.__id_count
        self.gen = gen
        self.mutation = mutation
        self.emote_layers = emote_layers
        self.num_emote_neurons = num_emote_neurons
        self.resp_layers = resp_layers
        self.num_resp_neurons = num_resp_neurons
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
        self.score = np.zeros(shape=(1))
        self.cu_score = 0
        self.output = np.zeros(shape=(1))
        self.input = np.zeros(shape=(1))
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
                self.emote_neuron[i][j][1] = self.silent_threshold
                self.emote_neuron[i][j][2] = self.active_threshold
                self.emote_neuron[i][j][3] = self.LTDLTP
                self.emote_neuron[i][j][4] = 0.5
                self.emote_neuron[i][j][5] = 0.5
                
                
                if i!=0:
                    ## Forward Connection from immediate neuron in previous layer
                    self.emote_neural_structure[i][j][0][j][0] = random.uniform(0.5,1.5)
                    ## Setting the default structure flag
                    self.emote_neural_structure[i][j][0][j][2] = 1
                    
                    if j<self.num_emote_neurons-1 :
                        if i<self.emote_layers-1:
                            ## Backward Connection from adjacent neuron in next layer
                            self.emote_neural_structure[i][j][1][j+1][0] = random.uniform(0.5,1.5)
                            ## Setting the default structure flag
                            self.emote_neural_structure[i][j][1][j+1][2] = 1
                        
                        ## Forward Connection from right adjacent neuron in previous layer
                        self.emote_neural_structure[i][j][0][j+1][0] = random.uniform(0.5,1.5)
                        ## Setting the default structure flag
                        self.emote_neural_structure[i][j][0][j+1][2] = 1
                else:
                    if j<self.num_emote_neurons-1 :
                        ## Backward Connection from adjacent neuron in next layer
                        self.emote_neural_structure[i][j][1][j+1][0] = random.uniform(0.5,1.5)
                        ## Setting the default structure flag
                        self.emote_neural_structure[i][j][1][j+1][2] = 1
                    ## Forward connection of the first layer
                    self.emote_neural_structure[i][j][0][0][0] = random.uniform(0.5,1.5)
                    ## Setting the default structure flag
                    self.emote_neural_structure[i][j][0][0][2] = 1
        
        ## Building structural connections for Response System
        for ii in range(self.resp_layers):

            for jj in range(self.num_resp_neurons):
                
                ## Setting neuron parameters
                self.response_neuron[ii][jj][1] = self.silent_threshold
                self.response_neuron[ii][jj][2] = self.active_threshold
                self.response_neuron[ii][jj][3] = self.LTDLTP
                self.response_neuron[ii][jj][4] = 0.5
                self.response_neuron[ii][jj][5] = 0.5

                if ii!=0:
                    ## Forward Connection from immediate neuron in previous layer
                    self.resp_neural_structure[ii][jj][0][jj][0] = random.uniform(0.5,1.5)
                    ## Setting the default structure flag
                    self.resp_neural_structure[ii][jj][0][jj][2] = 1
                    
                    if jj<self.num_resp_neurons-1:
                        if ii<self.resp_layers-1:
                            ## Backward Connection from adjacent neuron in next layer
                            self.resp_neural_structure[ii][jj][1][jj+1][0] = random.uniform(0.5,1.5)
                            ## Setting the default structure flag
                            self.resp_neural_structure[ii][jj][1][jj+1][2] = 1
                        
                        ## Forward Connection from right adjacent neuron in previous layer
                        self.resp_neural_structure[ii][jj][0][jj+1][0] = random.uniform(0.5,1.5)
                        ## Setting the default structure flag
                        self.resp_neural_structure[ii][jj][0][jj+1][2] = 1
                    #else:
                    #    if jj%2 == 0:
                            ## Implement output backpath
                    #        pass
                else:
                    if jj<self.num_resp_neurons-1:
                        ## Backward Connection from adjacent neuron in next layer
                        self.resp_neural_structure[ii][jj][1][jj+1][0] = random.uniform(0.5,1.5)
                        ## Setting the default structure flag
                        self.resp_neural_structure[ii][jj][1][jj+1][2] = 1
                    ## First position in first layer reserved for input 
                    self.resp_neural_structure[ii][jj][0][0][0] = random.uniform(0.5,1.5)
                    ## Setting the default structure flag
                    self.resp_neural_structure[ii][jj][0][0][2] = 1
                    for kk in range(self.num_emote_neurons):
                        ## Output of Primal Emotion system connected to response system
                        self.resp_neural_structure[ii][jj][0][kk+1][0] = random.uniform(0.5,1.5)
                        ## Setting the default structure flag
                        self.resp_neural_structure[ii][jj][0][kk+1][2] = 1
                        
    def random_connection(self):
        
        ## Building random connections for Primal Emotion System
        for i in range(self.emote_layers):

            for j in range(self.num_emote_neurons):
                ## Forward path
                for k in range(self.num_emote_neurons):
                    if i<self.emote_layers-1:
                        if self.emote_neural_structure[i][k][1][j][0] != 0:
                            self.emote_neural_structure[i+1][j][0][k][0] = 0
                        else:
                            if random.randint(0,self.emote_layers+1-i)>0:
                                self.emote_neural_structure[i+1][j][0][k][0] = random.uniform(0.5,1.5)
                ## Backward path
                for k in range(self.num_emote_neurons):
                    if i<self.emote_layers-1:
                        if self.emote_neural_structure[i+1][k][0][j][0] != 0:
                            self.emote_neural_structure[i][j][1][k][0] = 0
                        else:
                            if random.randint(0,self.emote_layers+1-i)>0:
                                self.emote_neural_structure[i][j][1][k][0] = random.uniform(0.5,1.5)

        ## Building random connections for Response System                        
        for ii in range(self.resp_layers):

            for jj in range(self.num_resp_neurons):
                ## Forward path
                for kk in range(self.num_resp_neurons):
                    if ii<self.resp_layers-1:
                        if self.resp_neural_structure[ii][kk][1][jj][0] != 0:
                            self.resp_neural_structure[ii+1][jj][0][kk][0] = 0
                        else:
                            if random.randint(0,self.resp_layers+1-ii)>0:
                                self.resp_neural_structure[ii+1][jj][0][kk][0] = random.uniform(0.5,1.5)
                ## Backward path
                for kk in range(self.num_resp_neurons):
                    if ii<self.resp_layers-1:
                        if self.resp_neural_structure[ii+1][kk][0][jj][0] != 0:
                            self.resp_neural_structure[ii][jj][1][kk][0] = 0
                        else:
                            if random.randint(0,self.resp_layers+1-i)>0:
                                self.resp_neural_structure[ii][jj][1][kk][0] = random.uniform(0.5,1.5)
                                
    def prim_emo_fire(self,input_val):
        #print('emo')
        self.emote_neuron[:,:,0] += self.neural_decay*self.emote_neuron[:,:,7]
        self.emote_neuron[:,:,7] = 0
        self.old_em = self.emote_copy
        for i in range(self.emote_layers):
            for j in range(self.num_emote_neurons):
                if self.emote_neuron[i,j,0] == 0:
                    self.emote_neuron[i,j,6] = 0
                else:
                    self.emote_neuron[i,j,6] = 1
                    if self.emote_neuron[i,j,0] < self.emote_neuron[i,j,1]:
                        self.emote_neuron[i,j,7] = self.emote_neuron[i,j,0]
                        self.emote_neuron[i,j,0] = 0
        
        self.emote_fire = np.dstack((self.emote_fire,self.emote_neuron[:,:,6]))
        
        for i in range(self.emote_layers-1,-1,-1):
            for j in range(self.num_emote_neurons-1,-1,-1):
                if i == 0:
                    neuro_fsum = np.tanh(self.emote_neural_structure[i,j,0,0,0]*input_val)
                    #print(self.emote_neural_structure[i,j,0,0,0]*input_val)
                    self.emote_copy[i,j,0,0] = self.emote_neural_structure[i,j,0,0,0]*input_val
                else:
                    neuro_fsum = np.sum(np.tanh(np.multiply\
                                                (self.emote_neural_structure[i,j,0,:,0],self.emote_neuron[i-1,:,0])))
                    #x=np.multiply(self.emote_neural_structure[i,j,0,:,0],\
                    #                                         self.emote_neuron[i-1,:,0])
                    #print(x)
                    self.emote_copy[i,j,0,:] = np.multiply(self.emote_neural_structure[i,j,0,:,0],\
                                                             self.emote_neuron[i-1,:,0])
                    
                if i < self.emote_layers-1:
                    neuro_bsum = np.sum(np.tanh(np.multiply\
                                                (self.emote_neural_structure[i,j,1,:,0],self.emote_neuron[i+1,:,0])))
                    self.emote_copy[i,j,1,:] = np.multiply(self.emote_neural_structure[i,j,1,:,0],\
                                                             self.emote_neuron[i+1,:,0])
                else:
                    neuro_bsum = 0

                if neuro_fsum + neuro_bsum <= 0:
                    self.emote_neuron[i,j,0] = 0
                else:
                    self.emote_neuron[i,j,0] = neuro_fsum + neuro_bsum
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
                
    def resp_sys_fire(self,input_val):
        #print('rs')
        #neuro_fsum = 0
        self.input = np.append(self.input,input_val)
        neuro_bsum = 0
        self.response_neuron[:,:,0] += self.neural_decay*self.response_neuron[:,:,7]
        self.response_neuron[:,:,7] = 0
        self.old_rs = self.resp_copy
        for i in range(self.resp_layers):
            for j in range(self.num_resp_neurons):
                if self.response_neuron[i,j,0] == 0:
                    self.response_neuron[i,j,6] = 0
                else:
                    self.response_neuron[i,j,6] = 1
                    if self.response_neuron[i,j,0] < self.response_neuron[i,j,2]:
                        self.response_neuron[i,j,7] = self.response_neuron[i,j,0]
                        self.response_neuron[i,j,0] = 0
        
        self.response_fire = np.dstack((self.response_fire,self.response_neuron[:,:,6]))
                    
        for i in range(self.resp_layers-1,-1,-1):
            for j in range(self.num_resp_neurons-1,-1,-1):
                if i == 0:
                    neuro_fsum = np.tanh(self.resp_neural_structure[i,j,0,0,0]*input_val)
                    neuro_fsum += np.sum(np.tanh
                                         (np.multiply(self.resp_neural_structure[i,j,0,1:7,0],self.emote_neuron[-1,:,0])))
                    self.resp_copy[i,j,0,1:7] = np.multiply(self.resp_neural_structure[i,j,0,1:7,0],\
                                                              self.emote_neuron[-1,:,0]) + \
                    (self.resp_neural_structure[i,j,0,0,0]*input_val)
                else:
                    neuro_fsum = np.sum(np.tanh
                                        (np.multiply(self.resp_neural_structure[i,j,0,:,0],self.response_neuron[i-1,:,0])))
                    self.resp_copy[i,j,0,:] = np.multiply(self.resp_neural_structure[i,j,0,:,0],\
                                                            self.response_neuron[i-1,:,0])
                    
                if i < self.resp_layers-1:    
                    neuro_bsum = np.sum(np.tanh
                                        (np.multiply(self.resp_neural_structure[i,j,1,:,0],self.response_neuron[i+1,:,0])))
                    self.resp_copy[i,j,1,:] = np.multiply(self.resp_neural_structure[i,j,1,:,0],\
                                                            self.response_neuron[i+1,:,0])
                
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
        self.output = np.append(self.output,np.sum(np.tanh(self.response_neuron[-1,:,0])))
        
        ## Score update
        self.score = np.append(self.score,(self.output[-2]-self.input[-1]))
        self.cu_score -= np.power(((self.output[-2]-self.input[-1])/self.input[-1]),2)
        ## Implement Synaptic space
        #self.prim_emo_control_system()
        #self.prim_emo_control_system()
    def prim_emo_control_system(self):
        ## To implement LTDLTP, weight update, excitory and inhibitory probability update
        ## Weight Update
        
        if len(self.output)>self.emote_layers:
            abssum = np.sum(np.absolute(self.emote_neuron[-1,:,0]))
            delta = (-self.lr * self.score[-1])*(1+self.emote_neuron[-1,0,0]/abssum)\
            *(1-self.emote_neuron[-1,1,0]/abssum)
            if np.isnan(delta):
                delta = self.lr * self.synaptic_diffuse * self.old_em
            else:
                delta *= self.synaptic_diffuse * self.old_em
            print('##########################################################################################')
            self.emote_neural_structure[:,:,:,:,0] = np.absolute(np.multiply(self.emote_neural_structure[:,:,:,:,0],1+delta))
            self.emote_neural_structure[:,:,:,:,0] += self.emote_neural_structure[:,:,:,:,2] + 0.01
            
            for i in range(self.emote_layers):

                for j in range(self.num_emote_neurons):
                    if np.max(self.emote_neural_structure[i,j,:,:,0]) > self.max_weight:
                        self.emote_neural_structure[i,j,:,:,0] /= np.max(self.emote_neural_structure[i,j,:,:,0])
            
            
            ## Fear or Ignore
            self.fear_ig = (-self.lr * self.score[-1])*(1+self.emote_neuron[-1,4,0]/abssum)\
            *(1-self.emote_neuron[-1,5,0]/abssum)
            
            if np.isnan(self.fear_ig):
                self.fear_ig = 0.2
            self.emote_neuron[:,:,2] *= 1 + self.fear_ig

            self.lr /= self.score[-1]
            ## Probability Update

            ## Synaptic scaling
    
    def resp_sys_control_system(self):
        ## To implement LTDLTP, weight update, excitory and inhibitory probability update
        ## Weight Update
        abssum = np.sum(np.absolute(self.emote_neuron[-1,:,0]))
        delta = (-self.lr * self.score[-1])*(1+self.emote_neuron[-1,0,0]/abssum)\
        *(1-self.emote_neuron[-1,1,0]/abssum)
        if np.isnan(delta):
            delta = self.lr * self.synaptic_diffuse * self.old_rs
        else:
            delta *= self.synaptic_diffuse * self.old_rs
        #print('##########################################################################################')
        
        self.resp_neural_structure[:,:,:,:,0] = np.absolute(np.multiply(self.resp_neural_structure[:,:,:,:,0],1+delta))
        self.resp_neural_structure[:,:,:,:,0] += self.resp_neural_structure[:,:,:,:,2] + 0.01
        
        for ii in range(self.resp_layers):

            for jj in range(self.num_resp_neurons):
                if np.max(self.resp_neural_structure[ii,jj,:,:,0]) > self.max_weight:
                    self.resp_neural_structure[ii,jj,:,:,0] /= np.max(self.resp_neural_structure[ii,jj,:,:,0])
                    self.resp_neural_structure[:,:,:,:,0] += self.resp_neural_structure[:,:,:,:,2] + 0.01
        ## Fear or Ignore
        
        #print(fear_ig)
        self.response_neuron[:,:,2] *= 1 + self.fear_ig
        
        ## Probability Update
    ''' 
    def save_model(self):
        
    def load_model(self):
       
    '''     
    def reset_id(self):
        SystemDesign.__id_count = 0
    
    def system_summary(self):

        print('id                                  : '+str(self.id))
        print('Generation                          : '+str(self.gen))
        print('Mutation Rate                       : '+str(self.mutation))
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
    

