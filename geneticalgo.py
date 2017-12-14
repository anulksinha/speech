from scipy.io import wavfile
from scipy import signal
#from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
plt.switch_backend('agg') 
#import systemdesign
import copy
from sysdesign import SystemDesign

if __name__ == '__main__':

    d1 = datetime.now()
    print('Time started : '+str(d1))
    filename = './dataLog/sample_4.wav'
    samplerate, data = wavfile.read(filename)
    times = np.arange(len(data))/float(samplerate)
    x = np.max(np.absolute(data))
    print(samplerate)
    print(x)
    print(data.shape)
    
    #print(data)
    #data_norm = normalize(data.reshape((-1,len(data)))).flatten()
    data_resam = np.multiply(1/x,data)
    print(data_resam.flatten().shape)
    data_resam[abs(data_resam[:])<0.0001] = 0.0001
    #data_resam += 1 
    
    #plt.plot(times,data_resam)
    #plt.show
    q = 0.5
    #print('resampling')
    #data_resam = signal.resample(data_norm,10000)
    #print('######################################')
    # Size of one sample set
    sam_pro = 10
    # Max number of generations
    Total_gen = 20
    # Max population per generation
    pop_gen = 20
    # Total sample sets for training
    totsam = 100
    # Threshold for stopping GA
    acc_thresh = 70
    # Number of input to be sent at a time to the network
    in_dim = 5
    # Emotion neurons for all models
    em_n = 12
    # Response Neurons for all models
    rs_n = 16
    # Emotion Layers for all models
    em_l = 3
    # Response Layers for all models
    rs_l = 4
    # Weight Update Active or Inactive
    w_update = True
    
    ## choose between mse optimization(True) and corrcoef optimization(False)
    
    mse_corr = True
    
    objlist = []
    list_sc = []
    best_score_gen = []
    dsam = True
    filename1 = './results/models/ModelBest.txt'
    filename2 = './results/models/Model2ndBest.txt'
    
    #print(data_resam[c*sam_pro:c*sam_pro+10])
    
    logfile = open('./dataLog/log.txt','w+')
    logfile.write('##################################################################################')
    logfile.write('\n#Time started                    : '+str(d1))
    logfile.write('\n#File                            : '+filename)
    logfile.write('\n#File Length                     :'+str(data.shape))
    logfile.write('\n#')
    logfile.write('\n#Sample set size                 : '+str(sam_pro))
    logfile.write('\n#Max generation per sample       : '+str(Total_gen))
    logfile.write('\n#Population per generation       : '+str(pop_gen))
    logfile.write('\n#Cross correlation threshold     : '+str(acc_thresh))
    logfile.write('\n#Total sample being trained on   : '+str(totsam))
    logfile.write('\n#Input Dimesion                  : '+str(in_dim))
    logfile.write('\n#Emotion Neurons                 : '+str(em_n))
    logfile.write('\n#Emotion Layers                  : '+str(em_l))
    logfile.write('\n#Response Neurons                : '+str(rs_n))
    logfile.write('\n#Response Layers                 : '+str(rs_l))
    logfile.write('\n#Weight Update Active            : '+str(w_update))
    logfile.write('\n#MSE(True) or Cross-Corr(False)  : '+str(mse_corr))
    logfile.write('\n##################################################################################')
    print('##################################################################################')
    print('Training Started')
    logfile.write('\nTraining Started')
    counter = 0
    
    for samp in range(totsam):
    #for g in range(Total_gen):
        objlist = []
        score = []
        list_sc = []
        while dsam:
            c = random.randint(0,len(data_resam)//sam_pro - 1)
            if sum(abs(data_resam[c*sam_pro:c*sam_pro+sam_pro])) <= 0.0001*sam_pro*10:
                c = random.randint(0,len(data_resam)//sam_pro - 1)
            else:
                dsam = False
        flag = True
        g = 0
        print('#############################################################')
        print('start sample : '+str(samp))
        while flag:
            #print(flag)
            print('\n#############################################################')
            print('\nstart sample : '+str(samp))
            print('\tGenetic Algorithm Generation : '+str(g))
            for s in range(pop_gen):
                if samp > 0:
                    new_net = copy.deepcopy(bestobj)
                    new_net.score = np.zeros(shape=(in_dim))
                    new_net.cu_score = 0
                    new_net.output = np.zeros(shape=(in_dim))
                    new_net.input = np.zeros(shape=(in_dim))
                    if s == 0:
                        new_net.reset_id()
                    new_net.gen = g+1
                    new_net.id = (g*pop_gen)+s+1 
                    for i in range(new_net.emote_layers):
                        for j in range(new_net.num_emote_neurons):
                            if np.random.rand() <= new_net.mutation:
                                x=random.randint(0,1)
                                p= random.randint(4,5)
                                new_net.emote_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)
                                new_net.emote_neuron[i,j,p] = random.uniform(0,1)
                    #print(new_net.resp_neural_structure[:,:,:,:,0])
                    
                    for i in range(4):
                        for j in range(new_net.num_emote_neurons+2):
                            if np.random.rand() <= new_net.mutation:
                                new_net.w[i,j] = np.random.uniform(-2,2)

                    for i in range(new_net.resp_layers):
                        for j in range(new_net.num_resp_neurons):
                            if np.random.rand() <= new_net.mutation:
                                x=random.randint(0,1)
                                p= random.randint(4,5)
                                new_net.resp_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)
                                new_net.response_neuron[i,j,p] = random.uniform(0,1)
                else:    
                    new_net = SystemDesign(gen=g+1,input_dim = in_dim,num_emote_neurons=em_n,\
                    resp_layers=rs_l,num_resp_neurons=rs_n, w_up = w_update, mse_corr=mse_corr)
                    new_net.structural_connection()
                    new_net.random_connection()
                objlist.append(new_net)
                #for obj in objlist:
                #    print(obj.gen,obj.id)
                #if g==0 and s==0 and samp>0:
                    #list_sc[-1].sort(key= lambda x:(x[1],x[2]))
                #    best_score_gen.append([counter-1,best_score,mse])
                ## Select top 2
                if g>0 and s == 0:
                    #print('ok g')
                    if mse_corr:
                        list_sc[-1].sort(key= lambda x:(x[2],x[1]))
                    else:
                        list_sc[-1].sort(key= lambda x:(x[1],x[2]))
                    print('\t'+str(list_sc[-1][-1]))
                    print('\t'+str(list_sc[-1][-2]))
                    best_score = list_sc[-1][-1][1]
                    mse = list_sc[-1][-1][2]
                    #print(list_sc[-1])
                    #print(scorelist)
                    print(g)
                    for obj in objlist:
                        #print(obj.id)
                        #print(list_sc[-1][-1][0])
                        #print(obj.gen,obj.id)
                        #if obj.gen == g:
                            
                            #print(obj.id)
                        if obj.id == list_sc[-1][-1][0]:
                            print(obj.id)
                            bestobj = copy.deepcopy(obj)
                            model1em = obj.emote_neural_structure
                            model1emne = obj.emote_neuron
                            model1rs = obj.resp_neural_structure
                            model1rsne = obj.response_neuron
                            model1w = obj.w
                            bestobj.save_model(filename=filename1)
                        elif obj.id == list_sc[-1][-2][0]:
                            print(obj.id)
                            best2obj = copy.deepcopy(obj)
                            model2em = obj.emote_neural_structure
                            model2emne = obj.emote_neuron
                            model2rs = obj.resp_neural_structure
                            model2rsne = obj.response_neuron
                            model2w = obj.w
                            best2obj.save_model(filename=filename2)
                    
                    print('\n\tBest and 2nd Best Models Saved\n')				
                    if g == Total_gen-1:
                        ## Save best models
                        bestobj.save_model(filename=filename1)
                        np.save('./results/models/EmNeuralBest',model1em)
                        np.save('./results/models/EmNeural2ndBest',model2em)
                        np.save('./results/models/RespNeuralBest',model1rs)
                        np.save('./results/models/RespNeural2ndBest',model2rs)
                        np.save('./results/models/EMNeuronBest',model1emne)
                        np.save('./results/models/EMNeuronBest2ndBest',model2emne)
                        np.save('./results/models/RSNeuronBest',model1rsne)
                        np.save('./results/models/RSNeuron2ndBest',model2rsne)
                        np.save('./results/models/WBest',model1w)
                        np.save('./results/models/W2ndBest',model2w)
                        print('\n\tBest and 2nd Best Models Saved\n')
                    ## Remove used elements
                    
                    #print('\t'+str(list_sc[-1]))
                    #del objlist[0:pop_gen]
                    best_score_gen.append([counter-1,best_score,mse])
                    #print(mse)
                    if mse_corr:
                        if abs(mse) < 0.001 or g>Total_gen:
                        #if best_score >= acc_thresh or mse < -0.001 or g>Total_gen:
                            print('Exceed')
                            flag = False
                            counter += 1
                            break
                            #print(flag)
                    else:
                        if best_score >= acc_thresh or g>Total_gen:
                            print('Exceed')
                            flag = False
                            counter += 1
                            break
                            #print(flag)

                

                ## Cross Breed
                if s %2 == 0 and g>0:
                    #print('ok s0')
                    
                    '''
                    new_net.emote_neural_structure[0,:,:,:,0] = model1em[0,:,:,:,0]
                    new_net.emote_neural_structure[1,:,:,:,0] = model2em[1,:,:,:,0]
                    new_net.emote_neural_structure[2,:,:,:,0] = model1em[2,:,:,:,0]

                    new_net.resp_neural_structure[0,:,:,:,0] = model1rs[0,:,:,:,0]
                    new_net.resp_neural_structure[1,:,:,:,0] = model2rs[1,:,:,:,0]
                    new_net.resp_neural_structure[2,:,:,:,0] = model1rs[2,:,:,:,0]
                    new_net.resp_neural_structure[3,:,:,:,0] = model2rs[3,:,:,:,0]
                    
                    new_net.emote_neuron[0,:,:] = model1emne[0,:,:]
                    new_net.emote_neuron[1,:,:] = model2emne[1,:,:]
                    new_net.emote_neuron[2,:,:] = model1emne[2,:,:]

                    new_net.response_neuron[0,:,:] = model1rsne[0,:,:]
                    new_net.response_neuron[1,:,:] = model2rsne[1,:,:]
                    new_net.response_neuron[2,:,:] = model1rsne[2,:,:]
                    new_net.response_neuron[3,:,:] = model2rsne[3,:,:]
                    '''
                    
                    
                    new_net.lr = bestobj.lr
                    new_net.epsilon = bestobj.epsilon
                    new_net.w[0,:] = model1w[0,:]
                    new_net.w[1,:] = model2w[1,:]
                    new_net.w[2,:] = model1w[2,:]
                    new_net.w[3,:] = model2w[3,:]

                    for i in range(new_net.emote_layers):
                        for j in range(new_net.num_emote_neurons):
                            if i%2 == 0:
                                new_net.emote_neural_structure[i,:,:,:,0] = model1em[i,:,:,:,0]
                                new_net.emote_neuron[i,:,:] = model1emne[i,:,:]
                            else:
                                new_net.emote_neural_structure[i,:,:,:,0] = model2em[i,:,:,:,0]
                                new_net.emote_neuron[i,:,:] = model2emne[i,:,:]
                            if np.random.rand() <= new_net.mutation:
                                x=random.randint(0,1)
                                p= random.randint(4,5)
                                new_net.emote_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)
                                new_net.emote_neuron[i,j,p] = random.uniform(0,1)
                    #print(new_net.resp_neural_structure[:,:,:,:,0])
                    
                    for i in range(new_net.resp_layers):
                        for j in range(new_net.num_resp_neurons):
                            if i%2 == 0:
                                new_net.resp_neural_structure[i,:,:,:,0] = model1rs[i,:,:,:,0]
                                new_net.response_neuron[i,:,:] = model1rsne[i,:,:]
                            else:
                                new_net.resp_neural_structure[i,:,:,:,0] = model2rs[i,:,:,:,0]
                                new_net.response_neuron[i,:,:] = model2rsne[i,:,:]
                            if np.random.rand() <= new_net.mutation:
                                x=random.randint(0,1)
                                p= random.randint(4,5)
                                new_net.resp_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)
                                new_net.response_neuron[i,j,p] = random.uniform(0,1)
                    
                    for i in range(4):
                        for j in range(new_net.num_emote_neurons+2):
                            if np.random.rand() <= new_net.mutation:
                                new_net.w[i,j] = np.random.uniform(-2,2)

                if s %2 == 1 and g>0:
                    #print('ok s1')
                    '''
                    new_net.emote_neural_structure[0,:,:,:,0] = model2em[0,:,:,:,0]
                    new_net.emote_neural_structure[1,:,:,:,0] = model1em[1,:,:,:,0]
                    new_net.emote_neural_structure[2,:,:,:,0] = model2em[2,:,:,:,0]

                    new_net.resp_neural_structure[0,:,:,:,0] = model2rs[0,:,:,:,0]
                    new_net.resp_neural_structure[1,:,:,:,0] = model1rs[1,:,:,:,0]
                    new_net.resp_neural_structure[2,:,:,:,0] = model2rs[2,:,:,:,0]
                    new_net.resp_neural_structure[3,:,:,:,0] = model1rs[3,:,:,:,0]
                    
                    new_net.emote_neuron[0,:,:] = model2emne[0,:,:]
                    new_net.emote_neuron[1,:,:] = model1emne[1,:,:]
                    new_net.emote_neuron[2,:,:] = model2emne[2,:,:]

                    new_net.response_neuron[0,:,:] = model2rsne[0,:,:]
                    new_net.response_neuron[1,:,:] = model1rsne[1,:,:]
                    new_net.response_neuron[2,:,:] = model2rsne[2,:,:]
                    new_net.response_neuron[3,:,:] = model1rsne[3,:,:]
                    '''
                    new_net.lr = best2obj.lr
                    new_net.epsilon = best2obj.epsilon
                    new_net.w[0,:] = model1w[0,:]
                    new_net.w[1,:] = model2w[1,:]
                    new_net.w[2,:] = model1w[2,:]
                    new_net.w[3,:] = model2w[3,:]
                    
                    
                    for i in range(new_net.emote_layers):
                        for j in range(new_net.num_emote_neurons):
                            if i%2 == 0:
                                new_net.emote_neural_structure[i,:,:,:,0] = model2em[i,:,:,:,0]
                                new_net.emote_neuron[i,:,:] = model2emne[i,:,:]
                            else:
                                new_net.emote_neural_structure[i,:,:,:,0] = model1em[i,:,:,:,0]
                                new_net.emote_neuron[i,:,:] = model1emne[i,:,:]
                            if np.random.rand() <= new_net.mutation:
                                x=random.randint(0,1)
                                new_net.emote_neural_structure[i,j,x,:,0] = random.uniform(0.5,1.5)
                    #print(new_net.resp_neural_structure[:,:,:,:,0])
                    
                    for i in range(4):
                        for j in range(new_net.num_emote_neurons+2):
                            if np.random.rand() <= new_net.mutation:
                                new_net.w[i,j] = np.random.uniform(-2,2)

                    for i in range(new_net.resp_layers):
                        for j in range(new_net.num_resp_neurons):
                            if i%2 == 0:
                                new_net.resp_neural_structure[i,:,:,:,0] = model2rs[i,:,:,:,0]
                                new_net.response_neuron[i,:,:] = model2rsne[i,:,:]
                            else:
                                new_net.resp_neural_structure[i,:,:,:,0] = model1rs[i,:,:,:,0]
                                new_net.response_neuron[i,:,:] = model1rsne[i,:,:]
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
                    new_net.resp_sys_fire(data_resam[c*sam_pro + d])
                    #new_net.control_system()
                    
                acc = np.around(np.nan_to_num(np.corrcoef(x*new_net.input[-1:-sam_pro:-1],x*new_net.output[-2:-sam_pro-1:-1]))[0,1]*100, decimals =2)
                #print(new_net.input[-1:-10:-1])
                #print(new_net.output[-1:-10:-1])
                
                #if np.sum(np.power(new_net.score[new_net.resp_layers:],2))/(sam_pro-new_net.resp_layers) <= 0.0001:
                #    acc = 100
                err = -np.sum(np.power(new_net.score[-1:-sam_pro+new_net.resp_layers-1:-1],2))/(sam_pro-new_net.resp_layers)
                #err = 100 - acc
                print('\t######################################')
                print('\tGeneration %i Id %i Cross Correlation Accuracy %f MSE %f'%(new_net.gen,new_net.id,acc,err))
                #print('\toutput'+str(new_net.output))
                #print('Accuracy : '+str(acc))
                #print(np.sum(np.power(new_net.score[new_net.resp_layers:],2))/(sam_pro-new_net.resp_layers))
                print('\t######################################')
                logfile.write('\n\tScore                               : '+str(acc))
                score.append([new_net.id,abs(acc),err])
            print('\t#######################################################################')
            #print(score)
            list_sc.append((score))
            g += 1
            counter += 1
        list_sc[-1].sort(key= lambda x:(x[1],x[2]))
        print('\t'+str(list_sc[-1][-1]))
        #best_score = list_sc[-1][-1][1]
        #mse = list_sc[-1][-1][2]
        #for obj in objlist:
        #    #if obj.gen == g:
        #        #print(obj.id)
        #    if obj.id == list_sc[-1][-1][0]:
        #        print(obj.id)
        #        bestobj = copy.deepcopy(obj)
        #        model1em = obj.emote_neural_structure
        #        model1emne = obj.emote_neuron
        #        model1rs = obj.resp_neural_structure
        #        model1rsne = obj.response_neuron
        #        model1w = obj.w
        #        bestobj.save_model(filename=filename1)
        #    elif obj.id == list_sc[-1][-2][0]:
        #        print(obj.id)
        #        best2obj = copy.deepcopy(obj)
        #        model2em = obj.emote_neural_structure
        #        model2emne = obj.emote_neuron
        #        model2rs = obj.resp_neural_structure
        #        model2rsne = obj.response_neuron
        #        model2w = obj.w
        #        best2obj.save_model(filename=filename2)
        print('Sample End')
        print('########################################################')
    print('Training Ended')    
    d2 = datetime.now()
    logfile.write('Time Ended : '+str(d1))
    logfile.write('#######################################################################')
    print('Code ran for : '+str(d2-d1))
    logfile.write('Code ran for : '+str(d2-d1))
    
    x=np.nan_to_num(np.array(best_score_gen))
    x_whole = np.nan_to_num(np.array(list_sc))
    np.save('./results/score',x)
    np.save('./results/AllScore',x_whole)
    fig1 = plt.figure()
    plt.plot(x[:,0],x[:,1])
    plt.xlabel('Generations')
    plt.ylabel('Cross Correlation Accuracy')
    plt.title('Genetic Algorithm Model Optimization')
    plt.ylim(0,110)
    plt.show(0)
    fig1.savefig('./results/AccGen.png')
	
    fig2 = plt.figure()
    plt.plot(x[:,0],x[:,2])
    plt.xlabel('Generations')
    plt.ylabel('MSE')
    plt.title('Genetic Algorithm Model Optimization')
    plt.ylim(-1,0.2)
    plt.show(0)
    fig2.savefig('./results/MSEGen.png')
    
    ###############################################################################################################
    ## Testing phase
    filenamet = './dataLog/sample_4.wav'
    sampleratet, datat = wavfile.read(filenamet)
    timest = np.arange(len(datat))/float(sampleratet)
    xt = np.max(np.absolute(datat))
    print(sampleratet)
    print(xt)
    print(datat.shape)
    #print(data)
    #data_norm = normalize(data.reshape((-1,len(data)))).flatten()
    data_resamt = np.multiply(1/xt,datat)
    data_resamt[abs(data_resamt[:])<0.0001] = 0.0001
    #data_resam += 1
    print(data_resamt.flatten().shape)
    ## total sample set to be tested
    duration = 1000
    aclist = []
    score = []
    print('#############################################################################')
    print('Test start')
    logfile.write('#############################################################################')
    logfile.write('Test start')
    logfile.write('\n##########################################################################################\n')
    logfile.write('Generation                          : '+str(bestobj.gen)+'\n')
    logfile.write('id                                  : '+(str(bestobj.id))+'\n')
    logfile.write('Epsilon                             : '+str(bestobj.epsilon))
    logfile.write('Learning Rate                       : '+str(bestobj.lr))
    logfile.write('Weight Update                       : '+str(bestobj.w))
    logfile.write('Mutation Rate                       : '+str(bestobj.mutation)+'\n')
    logfile.write('Primal Emotion layers               : '+str(bestobj.emote_layers)+'\n')
    logfile.write('Primal Emotion neurons per layer    : '+str(bestobj.num_emote_neurons)+'\n')
    logfile.write('Response system layers              : '+str(bestobj.resp_layers)+'\n')
    logfile.write('Response system neurons per layer   : '+str(bestobj.num_resp_neurons)+'\n')
    logfile.write('MAX Forward connections per neuron  : '+str(bestobj.f_max_connections)+'\n')
    logfile.write('MAX Backward connections per neuron : '+str(bestobj.b_max_connections)+'\n')
    logfile.write('MAX neural connection weight        : '+str(bestobj.max_weight)+'\n')

    logfile.write('\nPrimal Emotion System Connection  : \n'+str(bestobj.emote_neural_structure[:,:,:,:,0]))
    logfile.write('\nPrimal Emotion System Neuron      : \n'+str(bestobj.emote_neuron[:,:,0]))
    logfile.write('\nResponse System Connection        : \n'+str(bestobj.resp_neural_structure[:,:,:,:,0]))
    logfile.write('\nResponse System Neuron            : \n'+str(bestobj.response_neuron[:,:,0]))
    logfile.write('\nPrimal Emotion Neuron threshold   : \n'+str(bestobj.emote_neuron[:,:,2]))
    logfile.write('\nResponse Neuron threshold         : \n'+str(bestobj.response_neuron[:,:,2]))
    logfile.write('\nPrimal Emotion Neuron excite prob : \n'+str(bestobj.emote_neuron[:,:,4]))
    logfile.write('\nResponse Neuron excite prob       : \n'+str(bestobj.response_neuron[:,:,4]))
    logfile.write('\nPrimal Emotion Neuron inhibit prob: \n'+str(bestobj.emote_neuron[:,:,5]))
    logfile.write('\nResponse Neuron inhibit prob      : \n'+str(bestobj.response_neuron[:,:,5]))
    bestobj.system_summary()
    
    for i in range(duration):
        dsamt = True
        while dsamt:
            ct = random.randint(0,len(data_resamt)//sam_pro - 1)
            if sum(abs(data_resamt[ct*sam_pro:ct*sam_pro+sam_pro])) <= 0.0001*sam_pro:
                ct = random.randint(0,len(data_resamt)//sam_pro - 1)
            else:
                dsamt = False
        bestobj.score = np.zeros(shape=(in_dim))
        bestobj.cu_score = 0
        bestobj.output = np.zeros(shape=(in_dim))
        bestobj.input = np.zeros(shape=(in_dim))
        for dt in range(sam_pro):
              
            #if d%800 == 0:
                #print('######################################')
                #print('Generation %i Id %i'%(bestobj.gen,bestobj.id))
                #print('data %f %% processed'%((d+1)*100/len(data_resamt)))
                #print('%i ms data processed'%(d//8))
            bestobj.prim_emo_fire(data_resamt[ct*sam_pro + dt])
            bestobj.resp_sys_fire(data_resamt[ct*sam_pro + dt])
            #bestobj.control_system()
        acc = np.around(np.nan_to_num(np.corrcoef(xt*bestobj.input[-1:-sam_pro:-1],xt*bestobj.output[-2:-sam_pro-1:-1]))[0,1]*100, decimals =2)
        #if np.sum(np.power(bestobj.score[bestobj.resp_layers:],2))/(sam_pro-bestobj.resp_layers) <= 0.0001:
        #    acc = 100
        err = -np.sum(np.power(bestobj.score[-1:-sam_pro+bestobj.resp_layers-1:-1],2))/(sam_pro-bestobj.resp_layers)
        #err = 100 -acc		
        print('\t######################################')
        print('\tSample set : '+str(i))
        print('\tGeneration %i Id %i Cross Correlation Accuracy %f MSE %f'%(bestobj.gen,bestobj.id,acc,err))
        #print('Accuracy : '+str(acc))
        #print(np.sum(np.power(bestobj.score[bestobj.resp_layers:],2))/(sam_pro-bestobj.resp_layers))
        print('\t######################################')
        logfile.write('\n\tScore                               : '+str(acc))
        score.append([i+1,abs(acc),err])
        print('\t#######################################################################')
        #print(score)
        #aclist.append(score)
    bestobj.system_summary()
    print('Test End')
    print('#############################################################################')
    logfile.write('#############################################################################')
    logfile.write('Test End')
    logfile.write('\n##########################################################################################\n')
    logfile.write('Generation                          : '+str(bestobj.gen)+'\n')
    logfile.write('id                                  : '+(str(bestobj.id))+'\n')
    logfile.write('Epsilon                             : '+str(bestobj.epsilon))
    logfile.write('Learning Rate                       : '+str(bestobj.lr))
    logfile.write('Weight Update                       : '+str(bestobj.w))
    logfile.write('Mutation Rate                       : '+str(bestobj.mutation)+'\n')
    logfile.write('Primal Emotion layers               : '+str(bestobj.emote_layers)+'\n')
    logfile.write('Primal Emotion neurons per layer    : '+str(bestobj.num_emote_neurons)+'\n')
    logfile.write('Response system layers              : '+str(bestobj.resp_layers)+'\n')
    logfile.write('Response system neurons per layer   : '+str(bestobj.num_resp_neurons)+'\n')
    logfile.write('MAX Forward connections per neuron  : '+str(bestobj.f_max_connections)+'\n')
    logfile.write('MAX Backward connections per neuron : '+str(bestobj.b_max_connections)+'\n')
    logfile.write('MAX neural connection weight        : '+str(bestobj.max_weight)+'\n')

    logfile.write('\nPrimal Emotion System Connection  : \n'+str(bestobj.emote_neural_structure[:,:,:,:,0]))
    logfile.write('\nPrimal Emotion System Neuron      : \n'+str(bestobj.emote_neuron[:,:,0]))
    logfile.write('\nResponse System Connection        : \n'+str(bestobj.resp_neural_structure[:,:,:,:,0]))
    logfile.write('\nResponse System Neuron            : \n'+str(bestobj.response_neuron[:,:,0]))
    logfile.write('\nPrimal Emotion Neuron threshold   : \n'+str(bestobj.emote_neuron[:,:,2]))
    logfile.write('\nResponse Neuron threshold         : \n'+str(bestobj.response_neuron[:,:,2]))
    logfile.write('\nPrimal Emotion Neuron excite prob : \n'+str(bestobj.emote_neuron[:,:,4]))
    logfile.write('\nResponse Neuron excite prob       : \n'+str(bestobj.response_neuron[:,:,4]))
    logfile.write('\nPrimal Emotion Neuron inhibit prob: \n'+str(bestobj.emote_neuron[:,:,5]))
    logfile.write('\nResponse Neuron inhibit prob      : \n'+str(bestobj.response_neuron[:,:,5]))
    z=np.nan_to_num(np.array(score))
    #print(z)
    np.save('./results/testacc',z)

    fig3 = plt.figure()
    plt.plot(z[:,0],z[:,1])
    plt.xlabel('Samples')
    plt.ylabel('Test Cross Correlation Accuracy')
    plt.title('Genetic Algorithm Model Optimization')
    plt.ylim(0,110)
    plt.show(0)
    fig3.savefig('./results/AccGenTest.png')
    
    fig4 = plt.figure()
    plt.plot(z[:,0],z[:,2])
    plt.xlabel('Samples')
    plt.ylabel('Test MSE')
    plt.title('Genetic Algorithm Model Optimization')
    plt.ylim(-1,0.2)
    plt.show(0)
    fig4.savefig('./results/MSEGenTest.png')
    plt.show()
    
          