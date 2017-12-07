from scipy.io import wavfile
from scipy import signal
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta

from sysdesign import SystemDesign

d1 = datetime.now()
print('Time started : '+str(d1))
samplerate, data = wavfile.read('D:/sample_2.wav')
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
Total_gen = 400
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