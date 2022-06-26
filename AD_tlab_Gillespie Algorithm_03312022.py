#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 
import random
from scipy.integrate import odeint 


# In[2]:


E = [25]
S = [58]
ES= [5]
P = [13]

t = [0]
tend = 10

k1 = 0.5 
k2 = 1
k3 = 10 

while t[-1] < tend: 
    props = [ k1 * E[-1] * S[-1] , k2 * ES[-1] , k3* ES[-1] ]
    prop_sum = sum(props)
   
    if prop_sum == 0.0:
        break
    
    tau = np.random.exponential(scale=1/prop_sum)
    t.append(t[-1] +tau)
    
    rand = random.uniform(0,1)
    
    if rand * prop_sum <= props [0]: 
        E.append(E[-1] -1)
        S.append(S[-1] -1)
        ES.append(ES[-1] +1)
        P.append(P[-1])
        
    elif rand * prop_sum > props [0] and rand * prop_sum <= props [0] + props[1]:
        E.append(E[-1] +1)
        S.append(S[-1] +1)
        ES.append(ES[-1] -1)
        P.append(P[-1])
        
    elif rand * prop_sum > props [0] and rand * prop_sum <= props [0] + props[1] + props[2]:
        E.append(E[-1] +1)
        S.append(S[-1])
        ES.append(ES[-1] -1)
        P.append(P[-1])

E_plot = plt.plot(t,E, label = "E"),
S_plot = plt.plot(t,S, label = "S"),
ES_plot = plt.plot(t,ES, label = "ES"),
P_plot  = plt.plot(t,P, label = "P"),

#plt.legend(handles = [E_plot , S_plot, ES_plot, P_plot,])

plt.xlabel("Time")
plt.ylabel("Abundance")

plt.show ()


# In[ ]:


###ODE check

new_tend = t[-1]

y0 = [25,58,5,13]

t = np.linspace(0, new_tend, num = 1000)

params = [k1,k2,k3]

def sim(variables, t, params):
    E = variables [0]
    S =variables [1]
    ES=variables [2]
    P=variables [3]
    
    k1= params [0]
    k2= params [1]
    k2= params [2]
    

    dEdt = k2*ES+k3*ES-k1*E*S
    
    dSdt = k2*ES-k1*E*S
    
    dESdt = k1*E*S-k2*ES-k3*ES
    
    dPdt = k3*ES
    
    return([dEdt, dSdt, dEdt, dPdt])

y = odeint(sim, y0, t, args=(params,))

plt.plot(t,y[:,0])  #E
plt.plot(t,y[:,1])  #S
plt.plot(t,y[:,2])  #ES
plt.plot(t,y[:,3])  #P

plt.show()


# In[ ]:





# In[ ]:




