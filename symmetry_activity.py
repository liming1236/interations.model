from multiprocessing import Pool
import networkx as nx
from itertools import combinations
import string
import numpy as np
import random
import json
import pickle
import copy
from collections import OrderedDict
import pickle
import itertools
from multiprocessing import Pool
import random as rd
from scipy.stats import norm

class SimplagionModel():
    def __init__(self, G, node_neighbors_dict, triangles_list, l1, I_percentage, 
                       G2, node_neighbors_dict2, triangles_list2, l2, A_percentage):
        
        self.G = G
        self.neighbors_dict = node_neighbors_dict
        self.triangles_list = triangles_list
        self.l1 = l1
        self.nodes = list(node_neighbors_dict.keys())
        self.N = len(node_neighbors_dict.keys())
        self.I = int(I_percentage * self.N/100)
        self.G2 = G2
        self.neighbors_dict2 = node_neighbors_dict2
        self.triangles_list2 = triangles_list2
        self.l2 = l2
        self.nodes2 = list(node_neighbors_dict2.keys())
        self.N2 = len(node_neighbors_dict2.keys())
        self.A = int(A_percentage * self.N2/100)
        
        self.initial_infected_nodes = self.initial_setup()
        self.initial_infected_nodes2 = self.initial_setup2()
        
    def initial_setup(self, fixed_nodes_to_infect=None, print_status=True):

        self.sAgentSet = set()
        self.iAgentSet = set()
        self.iList = []       
        self.t = 0
        
        #start with everyone susceptible
        for n in self.nodes:
            self.sAgentSet.add(n)

        
        #infect nodes
        if fixed_nodes_to_infect==None: #the first time I create the model (the instance __init__)
            infected_this_setup=[]
            for ite in range(self.I): #we will infect I agents
                #select one to infect among the supsceptibles
                to_infect = random.choice(list(self.sAgentSet))
                self.infectAgent(to_infect)
                infected_this_setup.append(to_infect)
        else: #I already have run the model and this is not the first run, I want to infect the same nodes
            infected_this_setup=[]
            for to_infect in fixed_nodes_to_infect:
                self.infectAgent(to_infect)
                infected_this_setup.append(to_infect)
        return infected_this_setup
    
    def initial_setup2(self, fixed_nodes_to_infect2=None, print_status=True):
        #going to use this to store the agents in each state
        self.uAgentSet = set()
        self.aAgentSet = set()
        self.aList = []
        
        #start with everyone susceptible
        for n in self.nodes2:
            self.uAgentSet.add(n)
        
        #infect nodes
        if fixed_nodes_to_infect2==None:
            infected_this_setup2=[]
            for ite in range(self.A): #we will infect I agents
                #select one to infect among the supsceptibles
                to_infect2 = random.choice(list(self.uAgentSet))
                self.infectAgent2(to_infect2)
                infected_this_setup2.append(to_infect2)
        else: #I already have run the model and this is not the first run, I want to infect the same nodes
            infected_this_setup2=[]
            for to_infect2 in fixed_nodes_to_infect2:
                self.infectAgent2(to_infect2)
                infected_this_setup2.append(to_infect2)
        return infected_this_setup2
    
    def infectAgent(self,agent):
        self.iAgentSet.add(agent)
        self.sAgentSet.remove(agent)
        return 1
    
    def recoverAgent(self,agent):
        self.sAgentSet.add(agent)
        self.iAgentSet.remove(agent)
        return -1
    
    def infectAgent2(self,agent):
        self.aAgentSet.add(agent)
        self.uAgentSet.remove(agent)
        return 1
    
    def recoverAgent2(self,agent):
        self.uAgentSet.add(agent)
        self.aAgentSet.remove(agent)
        return -1
    
    def run(self, t_max, beta1, beta2, mu, beta3, beta4, mu2, theta1, theta2, eta1, eta2, print_status):
        
        self.t_max = t_max
        l1 = self.l1
        l2 = self.l2

        while len(self.iAgentSet) > 0 and len(self.sAgentSet) != 0 and self.t<=self.t_max:
            newIlist = set()
            newAlist = set()

            ####Information dissemination
            #Pairwise interaction propagation
            for aAgent in self.aAgentSet:
                for agent in self.neighbors_dict2[aAgent]:
                    if (agent in self.uAgentSet) and (l2[agent] + l2[aAgent] >= theta1):
                        if random.random() <= beta3: 
                            newAlist.add(agent)                           
            #Higher-order interaction propagation
            for triangle in self.triangles_list2:
                n1, n2, n3 = triangle
                if n1 in self.aAgentSet:
                    if n2 in self.aAgentSet:
                        if (n3 in self.uAgentSet) and (l2[n1] + l2[n2] + l2[n3] >= theta2):
                            if (random.random() <= beta4): 
                                newAlist.add(n3)
                    else:
                        if n3 in self.aAgentSet  and (l2[n1] + l2[n2] + l2[n3] >= theta2):
                            if (random.random() <= beta4): 
                                newAlist.add(n2)
                else:
                    if (n2 in self.aAgentSet) and (n3 in self.aAgentSet) and (l2[n1] + l2[n2] + l2[n3] >= theta2):
                        if (random.random() <= beta4): 
                            newAlist.add(n1) 

            ###Disease spread   
            #Pairwise interaction propagation       
            for iAgent in self.iAgentSet:
                for agent in self.neighbors_dict[iAgent]: 
                    if (agent in self.sAgentSet) and (agent in self.uAgentSet) and (l1[agent] + l1[iAgent] >= theta1):
                        if (random.random() <= beta1): 
                            newIlist.add(agent)
                    if (agent in self.sAgentSet) and (agent in self.aAgentSet) and (l1[agent] + l1[iAgent] >= theta1):
                        for i in self.neighbors_dict[agent]:
                            num1 = 0
                            if i in self.aAgentSet:
                                num1 += 1
                        gamma1 = num1/len(self.neighbors_dict[agent])+1     #1-simplex local awareness rate  

                        if (random.random() <= beta1*((1-gamma1)**eta1)): 
                            newIlist.add(agent)

            #Higher-order interaction propagation 
            node_count1 = {} # Initialize an empty dictionary to store the number of triangles each node is in           
            for triangle in self.triangles_list:
                for node in triangle:
                    if node in node_count1:
                        node_count1[node] += 1  
                    else:
                        node_count1[node] = 1  

            node_count2 = {} # Initialize an empty dictionary to store the number of awareness triangles each node is in
            for triangle in self.triangles_list:
                n1, n2, n3 = triangle
                if (n1 in self.aAgentSet) and (n2 in self.aAgentSet) and (n3 in self.aAgentSet):
                    for node in triangle:
                        if node in node_count2:
                            node_count2[node] += 1  
                        else:
                            node_count2[node] = 1                      
                 
            for triangle in self.triangles_list:
                n1, n2, n3 = triangle
                if n1 in self.iAgentSet:
                    if n2 in self.iAgentSet:
                        if (n3 in self.sAgentSet) and (n3 in self.uAgentSet) and (l1[n1] + l1[n2] + l1[n3] >= theta2):
                            if (random.random() <= beta2): 
                                newIlist.add(n3)
                        if  (n3 in self.sAgentSet) and (n3 in self.aAgentSet) and (l1[n1] + l1[n2] + l1[n3] >= theta2):
                            if n3 in node_count2:
                                gamma2 = node_count2[n3]/(node_count1[n3]+1)     #2-simplex local awareness rate of n3 
                            else:
                                gamma2 = 0                       
                            if (random.random() <= beta2*((1-gamma2)**eta2)): 
                                newIlist.add(n3)
                    else:
                        if n3 in self.iAgentSet and (n2 in self.uAgentSet) and (l1[n1] + l1[n2] + l1[n3] >= theta2):
                            if (random.random() <= beta2): 
                                newIlist.add(n2)
                        if n3 in self.iAgentSet and (n2 in self.aAgentSet) and (l1[n1] + l1[n2] + l1[n3] >= theta2):
                            if n2 in node_count2:
                                gamma2 = node_count2[n2]/(node_count1[n2]+1)     #2-simplex local awareness rate  of n2
                            else:
                                gamma2 = 0                                 
                            if (random.random() <= beta2*((1-gamma2)**eta2)): 
                                newIlist.add(n2)
                else:
                    if (n2 in self.iAgentSet) and (n3 in self.iAgentSet) and (n1 in self.uAgentSet) and (l1[n1] + l1[n2] + l1[n3] >= theta2):
                        if (random.random() <= beta2): 
                            newIlist.add(n1)
                    if (n2 in self.iAgentSet) and (n3 in self.iAgentSet) and (n1 in self.aAgentSet) and (l1[n1] + l1[n2] + l1[n3] >= theta2):
                        if n1 in node_count2:
                            gamma2 = node_count2[n1]/(node_count1[n1]+1)     #2-simplex local awareness rate  of n1
                        else:
                            gamma2 = 0                            
                        if (random.random() <= beta2*((1-gamma2)**eta2)): 
                            newIlist.add(n1)
     
            for n_to_infect in newAlist:
                self.infectAgent2(n_to_infect)
            for n_to_infect in newIlist:
                self.infectAgent(n_to_infect)
            
            
            #for recoveries
            newRlist = set()
            newRlist_A = set()
            if len(self.iAgentSet)<self.N:
            
                for recoverAgent in self.iAgentSet:
                    #if the agent has just been infected it will not recover this time
                    if recoverAgent in newIlist:
                        continue
                    else:
                        if (random.random() <= mu): 
                            newRlist.add(recoverAgent)

            #Update only now the nodes that have been infected
            for n_to_recover in newRlist:
                self.recoverAgent(n_to_recover)
            
            #then track the number of individuals in each state
            self.iList.append(len(self.iAgentSet))
            
            if len(self.aAgentSet)<self.N2:
            
                for recoverAgent in self.aAgentSet:
                    #if the agent has just been informed it will not recover this time
                    if recoverAgent in newAlist:
                        continue
                    else:
                        if (random.random() <= mu2): 
                            newRlist_A.add(recoverAgent)

            #Update only now the nodes that have been informed
            for n_to_recover in newRlist_A:
                self.recoverAgent2(n_to_recover)
            
            #then track the number of individuals in each state
            self.aList.append(len(self.aAgentSet))            
            #increment the time
            self.t += 1

        #and when we're done, return all of the relevant information
        if print_status: print('beta1', beta1, 'Done!', len(self.iAgentSet), 'infected agents left')

        return self.iList, self.aList
    
    def get_stationary_rho(self, normed=True, last_k_values = 100):
        i = self.iList
        if len(i)==0:
            return 0
        if normed:
            i = 1.*np.array(i)/self.N
        if i[-1]==1:
            return 1
        elif i[-1]==0:
            return 0
        else:
            avg_i = np.mean(i[-last_k_values:])
            avg_i = np.nan_to_num(avg_i) #if there are no infected left nan->0   
            return avg_i
        
    def get_stationary_rho2(self, normed=True, last_k_values = 100):
        i = self.aList
        if len(i)==0:
            return 0
        if normed:
            i = 1.*np.array(i)/self.N2
        if i[-1]==1:
            return 1
        elif i[-1]==0:
            return 0
        else:
            avg_i = np.mean(i[-last_k_values:])
            avg_i = np.nan_to_num(avg_i) #if there are no infected left nan->0   
            return avg_i
        
def run_one_simulation(args):
    
    it_num, N, p1, p2, lambda1s, lambdaD_targets, I_percentage, N2, p3, p4, lambda1s2, lambdaD_targets2, A_percentage, t_max, mu, mu2, theta1, theta2, eta1, eta2 = args #字符串变量名
    print('It %i initialized'%it_num)
    
    G, node_neighbors_dict, triangles_list, l1 = generate_my_simplicial_complex_d2(N,p1,p2)
    real_k = 1.*sum([len(v) for v  in node_neighbors_dict.values()])/len(node_neighbors_dict)
    real_kD = 3.*len(triangles_list)/len(node_neighbors_dict)

    G2, node_neighbors_dict2, triangles_list2, l2 = generate_my_simplicial_complex_d4(N2,p3,p4)
    real_k2 = 1.*sum([len(v) for v  in node_neighbors_dict2.values()])/len(node_neighbors_dict2)
    real_kD2 = 3.*len(triangles_list2)/len(node_neighbors_dict2)
    
    beta1s = 1.*(mu/real_k)*lambda1s
    beta2s = 1.*(mu/real_kD)*lambdaD_targets
    beta3s = 1.*(mu2/real_k2)*lambda1s2
    beta4s = 1.*(mu2/real_kD2)*lambdaD_targets2
    
    rhos = [] 
    rhos2 = []  
    beta3 = beta3s
    beta4 = beta4s

    for beta1 in beta1s:
        for beta2 in beta2s:
            mySimplagionModel = SimplagionModel(G, node_neighbors_dict, triangles_list, l1, I_percentage, 
                                                G2, node_neighbors_dict2, triangles_list2, l2, A_percentage)            
            mySimplagionModel.initial_setup(fixed_nodes_to_infect = mySimplagionModel.initial_infected_nodes);
            mySimplagionModel.initial_setup2(fixed_nodes_to_infect2 = mySimplagionModel.initial_infected_nodes2);
            results = mySimplagionModel.run(t_max, beta1, beta2, mu, beta3, beta4, mu2, theta1, theta2, eta1, eta2, print_status=False)
            rho = mySimplagionModel.get_stationary_rho(normed=True, last_k_values = 100)
            rhos.append(rho)
            rho2 = mySimplagionModel.get_stationary_rho2(normed=True, last_k_values = 100)
            rhos2.append(rho2)

    print('It %i, simulation has finished'%(it_num))

    return rhos, rhos2

    
def generate_my_simplicial_complex_d2(N,p1,p2):
    
    """Our model"""
    
    #I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, p1, seed=None)
    
    if not nx.is_connected(G):
        giant = list(nx.connected_components(G))[0]
        G = nx.subgraph(G, giant)
        print('not connected, but GC has order %i ans size %i'%(len(giant), G.size())) 

    triangles_list = []
    G_copy = G.copy()
    
    #Now I run over all the possible combinations of three elements:
    for tri in combinations(list(G.nodes()),3):
        #And I create the triangle with probability p2
        if random.random() <= p2:
            #I close the triangle.6 
            triangles_list.append(tri)
            
            #Now I also need to add the new links to the graph created by the triangle
            G_copy.add_edge(tri[0], tri[1])
            G_copy.add_edge(tri[1], tri[2])
            G_copy.add_edge(tri[0], tri[2])
            
    G = G_copy
             
    #Creating a dictionary of neighbors
    node_neighbors_dict = {}
    for n in list(G.nodes()):
        node_neighbors_dict[n] = list(G[n])        

    l1 = np.zeros(1000)
    nodes = list(G.nodes)    
    for node in nodes:
        l1[node] = np.clip(norm.rvs(loc=0.7, scale=0.3, size=1), 0, 1)     #Define the node activity distribution
              
    return G, node_neighbors_dict, triangles_list, l1

def generate_my_simplicial_complex_d4(N2,p3,p4):
    
    """Our model"""
    
    #I first generate a standard ER graph with edges connected with probability p1
    G2 = nx.fast_gnp_random_graph(N2, p3, seed=None)
    
    if not nx.is_connected(G2):
        giant = list(nx.connected_components(G2))[0]
        G2 = nx.subgraph(G2, giant)
        print('not connected, but G2C has order %i ans size %i'%(len(giant), G2.size())) 

    triangles_list2 = []
    G2_copy = G2.copy()
    
    #Now I run over all the possible combinations of three elements:
    for tri in combinations(list(G2.nodes()),3):
        #And I create the triangle with probability p2
        if random.random() <= p4:
            #I close the triangle.6 
            triangles_list2.append(tri)
            
            #Now I also need to add the new links to the graph created by the triangle
            G2_copy.add_edge(tri[0], tri[1])
            G2_copy.add_edge(tri[1], tri[2])
            G2_copy.add_edge(tri[0], tri[2])
            
    G2 = G2_copy
             
    #Creating a dictionary of neighbors
    node_neighbors_dict2 = {}
    for n in list(G2.nodes()):
        node_neighbors_dict2[n] = list(G2[n])    

    l2 = np.zeros(1000)
    nodes = list(G2.nodes)    
    for node in nodes:
        l2[node] = np.clip(norm.rvs(loc=0.7, scale=0.3, size=1), 0, 1)     #Define the node activity distribution

    return G2, node_neighbors_dict2, triangles_list2, l2

def get_p1_and_p2(k1,k2,N):
    p2 = (2.*k2)/((N-1.)*(N-2.))
    p1 = (k1 - 2.*k2)/((N-1.)- 2.*k2)
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')
    
def get_p3_and_p4(k3,k4,N2):
    p4 = (2.*k4)/((N2-1.)*(N2-2.))
    p3 = (k3 - 2.*k4)/((N2-1.)- 2.*k4)
    if (p3>=0) and (p4>=0):
        return p3, p4
    else:
        raise ValueError('Negative probability!')
            
def parse_results(results):    
    rhos_array, rhos2_array = [], []
    
    for rhos, rhos2 in results:
        rhos_array.append(rhos)
        rhos2_array.append(rhos2)
 
    rhos_array = np.array(rhos_array)  
    rhos2_array = np.array(rhos2_array)  
    avg_rhos = np.mean(rhos_array, axis=0)
    avg_rhos2 = np.mean(rhos2_array, axis=0)
    return avg_rhos, avg_rhos2

N = 1000                   
k1 = 20
k2 = 6

N2 = 1000                   
k3 = 20
k4 = 6
p1, p2 = get_p1_and_p2(k1,k2,N)
p3, p4 = get_p3_and_p4(k3,k4,N2)

theta1 = 1.0  
theta2 =  1.5
eta1 = 0.5 
eta2 = 0.5

mu = 0.05
lambda1s = np.linspace(0,2,30)  
lambdaD_targets = np.linspace(0,5,30)  

I_percentage = 1 #initial conditions (% of infected)

mu2 = 0.05
lambda1s2 = 1.5      
lambdaD_targets2 = 4
A_percentage = 1 #initial conditions (% of infected)

#Simulation Parameters
t_max = 2000                 
n_simulations = 120
n_processes = 8

out_dir = 'Results/'

iteration_numbers = range(n_simulations)

if __name__ == '__main__':
    
    args=[]
    for it_num in range(n_simulations):
        args.append([it_num, N, p1, p2, lambda1s, lambdaD_targets, I_percentage, N2, p3, p4, lambda1s2, lambdaD_targets2, A_percentage, t_max, mu, mu2, theta1, theta2, eta1, eta2])

    pool = Pool(processes=n_processes)                         
    results = pool.map(run_one_simulation, args)
    #Saving
    filename = "norm-" + "_loc(0.8)" +'N'+str(N) + '_lambda1s2' + str(lambda1s2) + '_lambdaD_targets2' + str(lambdaD_targets2) +'_theta1'+str(theta1) +'_theta2'+str(theta2) +'_seed'+str(I_percentage) +'.p'
    pickle.dump(results, open(out_dir+filename, "wb" ))
