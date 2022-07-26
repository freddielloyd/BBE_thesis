#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:50:55 2022

@author: freddielloyd
"""

import numpy as np
import random as rd
import copy

import matplotlib.pyplot as plt
import numpy as np

class graph:
    
    """Graph ADT"""    
    def __init__(self):
        self.graph={}
        self.visited={}
            
    def append(self,vertexid,edge,weight):        
        """add/update new vertex,edge,weight"""        
        if vertexid not in self.graph.keys():      
            self.graph[vertexid]={}
            self.visited[vertexid]=0
        if edge not in self.graph.keys():      
            self.graph[edge]={}
            self.visited[edge]=0
        self.graph[vertexid][edge]=weight
        
    def reveal(self):
        """return adjacent list"""
        return self.graph
    
    def vertex(self):
        """return all vertices in the graph"""
        return list(self.graph.keys())
    
    def edge(self,vertexid):
        """return edge of a particular vertex"""
        return list(self.graph[vertexid].keys())
    
    def edge_reverse(self,vertexid):
        """return vertices directing to a particular vertex"""                
        return [i for i in self.graph if vertexid in self.graph[i]]
    
    def weight(self,vertexid,edge):
        """return weight of a particular vertex"""
        return (self.graph[vertexid][edge])
    
    def order(self):
        """return number of vertices"""
        return len(self.graph)
    
    def visit(self,vertexid):
        """visit a particular vertex"""
        self.visited[vertexid]=1
        
    def go(self,vertexid):
        """return the status of a particular vertex"""
        return self.visited[vertexid]
    
    def route(self):
        """return which vertices have been visited"""
        return self.visited
    
    def degree(self,vertexid):
        """return degree of a particular vertex"""
        return len(self.graph[vertexid])
    
    def mat(self):
        """return adjacent matrix"""        
        self.matrix=[[0 for _ in range(max(self.graph.keys())+1)] for i in range(max(self.graph.keys())+1)]        
        for i in self.graph:    
            for j in self.graph[i].keys():    
                self.matrix[i][j]=1        
        return self.matrix
    
    def remove(self,vertexid):  
        """remove a particular vertex and its underlying edges"""
        for i in self.graph[vertexid].keys():
            self.graph[i].pop(vertexid)
        self.graph.pop(vertexid)
        
    def disconnect(self,vertexid,edge):
        """remove a particular edge"""
        del self.graph[vertexid][edge]
    
    def clear(self,vertexid=None,whole=False):
        """unvisit a particular vertex"""
        if whole:
            self.visited=dict(zip(self.graph.keys(),[0 for i in range(len(self.graph))]))
        elif vertexid:
            self.visited[vertexid]=0
        else:
            assert False,"arguments must satisfy whole=True or vertexid=int number"
            
            
            
            
wsmodel = graph()
            


#initial parameters
num_of_v=200
num_of_neighbors=60
prob=0.3

assert num_of_neighbors%2==0,"number of neighbors must be even number"


#first we create a regular ring lattice
for i in range(num_of_v):
    for j in range(1,num_of_neighbors//2+1):
        wsmodel.append(i,i+j if i+j<num_of_v else i+j-num_of_v,1)   
        wsmodel.append(i,i-j if i-j>=0 else i-j+num_of_v,1)
        wsmodel.append(i+j if i+j<num_of_v else i+j-num_of_v,i,1)   
        wsmodel.append(i-j if i-j>=0 else i-j+num_of_v,i,1)
        
        
        
        
#rewiring
#remove a random edge and create a random edge
for i in wsmodel.vertex():
    for j in wsmodel.edge(i):
        if np.random.uniform()<prob:
            wsmodel.disconnect(i,j)
            wsmodel.disconnect(j,i)
            rewired=np.random.choice(wsmodel.vertex())
            wsmodel.append(i,rewired,1)
            wsmodel.append(rewired,i,1)
            
            
            
# ##### Degree Distribution

#get degree
degree_dst=[wsmodel.degree(node) for node in wsmodel.vertex()]

#viz
ax=plt.figure(figsize=(10,5)).add_subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.hist(degree_dst,bins=30,width=0.7,color='#124fcc')
plt.title('Degree Distribution of Watts-Strogatz Model')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()



