import networkx as nx
import numpy as np
import random as rd
import numpy as np
import json
import random
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
import pandas as pd
from scipy.stats import norm

#model
def markovChain(G, G2, l1, l2, beta, beta_D, betas2, beta_D2, mu, mu2, theta1, theta2, eta1, eta2, node_neighbors_dict, node_neighbors_dict2, tri_neighbors_dict, triangles_list2, NSteps, i0):

    N = len(node_neighbors_dict)    
    p = np.zeros(2000)
    p[rd.sample(list(G.nodes),int(N*i0))] = 1
    p_new = np.copy(p)
    
    q = 1
    pTime = [np.mean(p)*2000/N]
    for k in range(0,NSteps):
    #Defining the state of nodes in the information layer
        uAgentSet = set()
        aAgentSet = set()
        for i in G2.nodes():
            if random.random() < 0.1:
                aAgentSet.add(i)  #Aware state
            else:
                uAgentSet.add(i)  #unaware state

        #####Information dissemination
        #Pairwise interaction propagation
        for i in aAgentSet.copy():
            for j in node_neighbors_dict2[i]:
                if (j in uAgentSet) and (l2[j] + l2[i] >= theta1):
                    if random.random() <= betas2: 
                        aAgentSet.add(j)                           
        #Higher-order interaction propagation
        for triangle in triangles_list2:
            n1, n2, n3 = triangle
            if n1 in aAgentSet:
                if n2 in aAgentSet:
                    if (n3 in uAgentSet) and (l2[n1] + l2[n2] + l2[n3] >= theta2):
                        if (random.random() <= beta_D2): 
                            aAgentSet.add(n3)
                else:
                    if n3 in aAgentSet  and (l2[n1] + l2[n2] + l2[n3] >= theta2):
                        if (random.random() <= beta_D2): 
                            aAgentSet.add(n2)
            else:
                if (n2 in aAgentSet) and (n3 in aAgentSet) and (l2[n1] + l2[n2] + l2[n3] >= theta2):
                    if (random.random() <= beta_D2): 
                        aAgentSet.add(n1) 

        for i in aAgentSet.copy():
            if random.random() < mu2:
                aAgentSet.remove(i)

        for i in G2.nodes():
            if i in aAgentSet:
                G2.nodes[i]['as'] = 'A' #Aware state
            else:
                G2.nodes[i]['as'] = 'U' #unaware state

        ###Disease spread                         
        for i in list(G.nodes):
            if G2.nodes[i]['as'] == 'U' :
                #Pairwise interaction propagation
                for j in node_neighbors_dict[i]:
                    if l1[i] + l1[j] >= theta1:                    
                        q *= (1.-beta*p[j])
                    
                #Higher-order interaction propagation
                for j, k in tri_neighbors_dict[i]:
                    if l1[i]+l1[j]+l1[k] >= theta2:
                        q *= (1.-beta_D*p[j]*p[k])
                
                #Updating the vector
                p_new[i] = (1-q)*(1-p[i]) + (1.-mu)*p[i]

            if G2.nodes[i]['as'] == 'A' :
                #Pairwise interaction propagation
                for j in node_neighbors_dict[i]:
                    if l1[i]+l1[j] >= theta1:
                        num1 = 0
                        if j in aAgentSet:
                            num1 += 1
                        gamma1 = num1/len(node_neighbors_dict[i])+1          #1-simplex local awareness rate                                    
                        q *= (1.-(1-gamma1)**eta1*beta*p[j])
                    
                #Higher-order interaction propagation
                node_count1 = {} # Initialize an empty dictionary to store the number of triangles each node is in           
                for triangle in triangles_list:
                    for node in triangle:
                        if node in node_count1:
                            node_count1[node] += 1  
                        else:
                            node_count1[node] = 1  

                node_count2 = {} # Initialize an empty dictionary to store the number of awareness triangles each node is in
                for triangle in triangles_list:
                    n1, n2, n3 = triangle
                    if (n1 in aAgentSet) and (n2 in aAgentSet) and (n3 in aAgentSet):
                        for node in triangle:
                            if node in node_count2:
                                node_count2[node] += 1  
                            else:
                                node_count2[node] = 1   

                for j, k in tri_neighbors_dict[i]:
                    if l1[i]+l1[j]+l1[k] >= theta2:
                        if i in node_count2:
                            gamma2 = node_count2[i]/(node_count1[i]+1)     #2-simplex local awareness rate   
                        else:
                            gamma2 = 0
                        q *= (1.-(1-gamma2)**eta2*beta_D*p[j]*p[k])       

                #Updating the vector
                p_new[i] = (1-q)*(1-p[i]) + (1.-mu)*p[i]

            #Resetting the i-th parameters
            q = 1
            
        p = np.copy(p_new)
        pTime.append(np.mean(p)*2000/N)
    return np.mean(pTime[int(NSteps*0.8):])

def get_tri_neighbors_dict(triangles_list):
    tri_neighbors_dict = defaultdict(list)
    for i, j, k in triangles_list:
        tri_neighbors_dict[i].append((j,k))
        tri_neighbors_dict[j].append((i,k))
        tri_neighbors_dict[k].append((i,j))
    return tri_neighbors_dict

def import_sociopattern_simcomp_SCM(dataset_dir, dataset, n_minutes, thr):
    filename = dataset_dir+'random_'+str(n_minutes)+'_'+str(thr)+'min_cliques_'+dataset+'.json'
    SCM_cliques_list = json.load(open(filename,'r'))
    
    #considering one realization of the SCM
    realization_number = random.choice(range(len(SCM_cliques_list)))
    cliques = SCM_cliques_list[realization_number] 
    G, l1, node_neighbors_dict, triangles_list = create_simplicial_complex_from_cliques(cliques)
    
    N = len(node_neighbors_dict.keys())
    avg_k1 = 1.*sum([len(v) for v in node_neighbors_dict.values()])/N
    avg_k2 = 3.*len(triangles_list)/N 
    
    return G, l1, node_neighbors_dict, triangles_list, avg_k1, avg_k2

def import_sociopattern_simcomp_SCM2(dataset_dir, dataset, n_minutes, thr):
    filename = dataset_dir+'random_'+str(n_minutes)+'_'+str(thr)+'min_cliques_'+dataset+'.json'
    SCM_cliques_list = json.load(open(filename,'r'))
    
    #considering one realization of the SCM
    realization_number = random.choice(range(len(SCM_cliques_list)))
    cliques = SCM_cliques_list[realization_number] 
    G2, l2, node_neighbors_dict2, triangles_list2 = create_simplicial_complex_from_cliques2(cliques)
    
    N2 = len(node_neighbors_dict2.keys())
    avg_k1_2 = 1.*sum([len(v) for v in node_neighbors_dict2.values()])/N2
    avg_k2_2 = 3.*len(triangles_list2)/N2 
    
    return G2, l2, node_neighbors_dict2, triangles_list2, avg_k1_2, avg_k2_2

def create_simplicial_complex_from_cliques(cliques):
    
    G = nx.Graph()
    triangles_list = set() #will contain list of triangles (2-simplices)
    
    for c in cliques:
        d = len(c)
        
        if d==2:
            i, j = c
            G.add_edge(i, j)
        
        elif d==3:
            #adding the triangle as a sorted tuple (so that we don't get both ijk and jik for example)
            triangles_list.add(tuple(sorted(c)))
            #adding the single links
            for i, j in combinations(c, 2):
                G.add_edge(i, j)
            
        else: #d>3, but I only consider up to dimension 3
            #adding the triangles
            for i, j, k in combinations(c, 3):
                triangles_list.add(tuple(sorted([i,j,k])))

            #adding the single links
            for i, j in combinations(c, 2):
                G.add_edge(i, j)
                
    if nx.is_connected(G)==False:
        print('not connected')
                
    #Creating a dictionary of neighbors
    node_neighbors_dict = {}
    for n in G.nodes():
        node_neighbors_dict[n] = G[n].keys()

    triangles_list = [list(tri) for tri in triangles_list]

    l1 = np.zeros(len(G.nodes))
    nodes = list(G.nodes)    
    for node in nodes:
        l1[node] = np.clip(norm.rvs(loc=0.8, scale=0.3, size=1), 0, 1)   

    return  G, l1, node_neighbors_dict, triangles_list

def create_simplicial_complex_from_cliques2(cliques):
    
    G2 = nx.Graph()
    triangles_list2 = set() #will contain list of triangles (2-simplices)
    
    for c in cliques:
        d = len(c)
        
        if d==2:
            i, j = c
            G2.add_edge(i, j)
        
        elif d==3:
            #adding the triangle as a sorted tuple (so that we don't get both ijk and jik for example)
            triangles_list2.add(tuple(sorted(c)))
            #adding the single links
            for i, j in combinations(c, 2):
                G2.add_edge(i, j)
            
        else: #d>3, but I only consider up to dimension 3
            #adding the triangles
            for i, j, k in combinations(c, 3):
                triangles_list2.add(tuple(sorted([i,j,k])))

            #adding the single links
            for i, j in combinations(c, 2):
                G2.add_edge(i, j)
                
    if nx.is_connected(G2)==False:
        print('not connected')
                
    #Creating a dictionary of neighbors
    node_neighbors_dict2 = {}
    for n in G2.nodes():
        node_neighbors_dict2[n] = G2[n].keys()

    triangles_list2 = [list(tri) for tri in triangles_list2]

    l2 = np.zeros(len(G2.nodes))
    nodes = list(G2.nodes)    
    for node in nodes:
        l2[node] = np.clip(norm.rvs(loc=0.8, scale=0.3, size=1), 0, 1)   
    
    G2 = G.copy()
    return  G2, l2, node_neighbors_dict2, triangles_list2

# Reading clean Sociopatterns data
dataset_dir = 'Data/Sociopatterns/thr_data_random/'
dataset = 'Thiers13' #'InVS15','SFHH', 'LH10','LyonSchool','Thiers13'
n_minutes = 5 #Aggregation time
thr = 0.8 #fraction of removed cliques (0.80: retaining the 20% most weighted)

G, l1, node_neighbors_dict, triangles_list, avg_k1, avg_k2 = import_sociopattern_simcomp_SCM(dataset_dir, dataset, n_minutes, thr)
G2, l2, node_neighbors_dict2, triangles_list2, avg_k1_2, avg_k2_2 = import_sociopattern_simcomp_SCM2(dataset_dir, dataset, n_minutes, thr)
tri_neighbors_dict = get_tri_neighbors_dict(triangles_list)

mu = 0.05
lambda1s = np.linspace(0,2,30)
lambdaD_targets = 3
I_percentage = 5 #initial conditions (% of infected)

betas = 1.*(mu/avg_k1)*lambda1s
beta_D = 1.*(mu/avg_k2)*lambdaD_targets

mu2 = 0.05
lambda1s2 = 1
lambdaD_targets2 = 3

betas2 = 1.*(mu2/avg_k1_2)*lambda1s2
beta_D2 = 1.*(mu/avg_k2_2)*lambdaD_targets2

theta1 = 1
theta2 = 2

NSteps = 500
eta1 = 1
eta2 = 1

# Running Makov and saving it to a single list of lists
markov_results = []

for beta in betas:
    i0 = I_percentage/100.
    rho_markov = [markovChain(G, G2, l1, l2, beta, beta_D, betas2, beta_D2, mu, mu2, theta1, theta2, eta1, eta2, node_neighbors_dict, node_neighbors_dict2, tri_neighbors_dict, triangles_list2, NSteps, i0)]
    # print(rho_markov)
    markov_results.append(rho_markov)
data = np.array(markov_results)
df = pd.DataFrame(data)
df = df.fillna(0)
# print(df)
np.savetxt('Highschool-0.8.txt',(df))