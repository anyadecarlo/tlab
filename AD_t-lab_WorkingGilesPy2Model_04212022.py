#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy
import matplotlib.pyplot as plt
import gillespy2
from gillespy2 import Model, Species, Parameter, Reaction
from gillespy2.solvers.numpy.basic_ode_solver import ODESolver
from numpy import *
import pylab as p
import numpy as np
import pandas as pd
from numpy import array
import csv


# In[97]:


class LotkaVolterra(gillespy2.Model):

     # Model setup
     def __init__(self):

          # Superclass initialization
          gillespy2.Model.__init__(self, name = "Lotka-Volterra")

          # Species
          predator = gillespy2.Species(name = "predator", initial_value = 125)
          prey = gillespy2.Species(name = "prey", initial_value = 375)
          food = gillespy2.Species(name = "food", initial_value = 1)
          predator_dead = gillespy2.Species(name = "dead_predator", initial_value = 0)
          self.add_species([predator, prey, food, predator_dead])

          # Parameters (rates)
          r_prey_reproduction = gillespy2.Parameter(name = "r_prey_reproduction", expression = 8)
          r_predation = gillespy2.Parameter(name = "r_predation", expression = 0.02)
          r_predator_death = gillespy2.Parameter(name = "r_predator_death", expression =8)
          self.add_parameter([r_prey_reproduction, r_predation, r_predator_death])

          # Reactions
          prey_reproduction = gillespy2.Reaction(name = "prey_reproduction",
                                                 reactants = {food: 1, prey: 1},
                                                 products = {food: 1, prey: 2},
                                                 rate = r_prey_reproduction)
          predation = gillespy2.Reaction(name = "predation",
                                         reactants = {predator: 1, prey: 1},
                                         products = {predator: 2, prey: 0},
                                         rate = r_predation)
          predator_death = gillespy2.Reaction(name = "predator_death",
                                              reactants = {predator: 1},
                                              products = {predator_dead: 1},
                                              rate = r_predator_death)
          self.add_reaction([prey_reproduction, predation, predator_death])

          # Set default timespan
          self.timespan(np.linspace(0, 10, 500))

          # Set list of species that should be plotted
          self.species_to_plot = ["predator", "prey", "food"]


# In[98]:


model = LotkaVolterra()


# In[99]:


results = model.run()
#put results in a csv file 


# In[100]:


results.plot(yscale='log',xaxis_label='Time', yaxis_label="Value")


# In[101]:


results


# In[102]:


############################################
############################################
##turning results into a csv file##
###########################################
###########################################


# In[103]:


data = []
data = results

test = data[0]

#test = test1.replace("array","numpy.array")
import csv
# create a csv file  test.csv and store
# it in a variable as outfile
with open("test.csv", "w") as outfile:
 
    # pass the csv file to csv.writer function.
    writer = csv.writer(outfile)
 
    # pass the dictionary keys to writerow
    # function to frame the columns of the csv file
    writer.writerow(test.keys())
   
    # make use of writerows function to append
    # the remaining values to the corresponding
    # columns using zip function.
    writer.writerows(zip(*test.values()))


# In[104]:


#data[0]


# In[105]:


#results[0]


# In[106]:


#print(results)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




