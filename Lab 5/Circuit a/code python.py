import plotly.express as px
import pandas as pd
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar
import statistics
from tabulate import tabulate
import math

#df = pd.read_csv(r'C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit a\Mesure_resistance_1.csv')
#print(df.to_string(header=False))

#print("RES CIRCUIT 1#################################################################################")
#RÉSISTANCE CIRCUIT 1
res_1=[]
inc_res_1=[]
for i in range(10):
    if i == 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit a"
        #txt = str(i)
        name = r"\Mesure_resistance.csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        res_1.append(moyenne)
        inc_res_1.append(incertitude)
    if i != 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit a"
        txt = str(i)
        name = r"\Mesure_resistance_"+txt+".csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        res_1.append(moyenne)
        inc_res_1.append(incertitude)

#################################################################################################
#print("TENSION CIRCUIT 1######################")
#TENSION CIRCUIT 1
tens_1=[]
inc_tens_1=[]
for i in range(10):
    if i == 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit a"
        #txt = str(i)
        name = r"\Mesure_Tension_VI.csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        tens_1.append(moyenne)
        inc_tens_1.append(incertitude)
    if i != 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit a"
        txt = str(i)
        name = r"\Mesure_Tension_VI_"+txt+".csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        tens_1.append(moyenne)
        inc_tens_1.append(incertitude)

#print("PUISSANCE CIRCUIT 1########################")
puiss1=[]
for i in range(10):
    P = tens_1[i]*tens_1[i]/res_1[i]
    puiss1.append(P*1000)
    incertitude_puiss= P*math.sqrt((inc_res_1[i]/res_1[i])**2+((tens_1[i]**2*math.sqrt(2*(inc_tens_1[i]/tens_1[i])**2))/tens_1[i]**2)**2)
    #print(P,",",incertitude_puiss)


#################################################################################################
#print("RÉSISTANE CIRCUIT 2######################")
#RÉSISTANCE CIRCUIT 2
res_2=[]
inc_res_2=[]
for i in range(10):
    if i == 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit b cond"
        #txt = str(i)
        name = r"\Mesure_resistance_condensateur.csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        res_2.append(moyenne)
        inc_res_2.append(incertitude)
    if i != 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit b cond"
        txt = str(i)
        name = r"\Mesure_resistance_condensateur_"+txt+".csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        res_2.append(moyenne)
        inc_res_2.append(incertitude)

#################################################################################################
#print("TENSION CIRCUIT 2######################")
#TENSION CIRCUIT 2
tens_2=[]
inc_tens_2=[]
for i in range(10):
    if i == 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit b cond"
        #txt = str(i)
        name = r"\Mesure_Tension_VI_condensateur.csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude)
        tens_2.append(moyenne)
        inc_tens_2.append(incertitude)
    if i != 0:
        path = r"C:\Users\mathi\OneDrive\Bureau\Fin 2 ans bacc physique\Session 4\Physique expérimentale\Lab 5\Traitement de données\Lab 5\Circuit b cond"
        txt = str(i)
        name = r"\Mesure_Tension_VI_condensateur_"+txt+".csv"
        filename = path+name

        df = pd.read_csv(filename)
        data = df.Value

        moyenne = sum(data)/len(data)
        ecart_type = statistics.stdev(data)
        incertitude = ecart_type/len(data) 
        #print(moyenne,",", incertitude) 
        tens_2.append(moyenne)
        inc_tens_2.append(incertitude)  

#print("PUISSANCE CIRCUIT 2########################")
puiss2=[]
for i in range(10):
    P = tens_2[i]*tens_2[i]/res_2[i]
    puiss2.append(P*1000)
    incertitude_puiss= P*math.sqrt((inc_res_2[i]/res_2[i])**2+((tens_2[i]**2*math.sqrt(2*(inc_tens_2[i]/tens_2[i])**2))/tens_2[i]**2)**2)
    #print(P,",",incertitude_puiss)  



#on trouve les paramètres qui vons optimiser le curve fitting en connaissant la fonction que suit notre distribution
#sans condensateur
def func_p1(a,b,c):
    return ((a*(b)**2)/(c+a)**2)*1000

constants = curve_fit(func_p1, res_1, puiss1)
b_fit1 = constants[0][0]
c_fit1 = constants[0][1]

#on trace la fonction avec ces paramètres
def func_fitted1(x):
    return ((x*b_fit1**2)/(c_fit1+x)**2)*1000
x = np.linspace(10**0,10**3,1000)


#Avec condensateur
def func_p2(a,b,c,d):
    return (a*b**2)/((c+a)**2+(d+39.7887)**2)*1000

constants = curve_fit(func_p2, res_2, puiss2)
b_fit2 = constants[0][0]
c_fit2 = constants[0][1]
d_fit2 = constants[0][2]

#on trace la fonction avec ces paramètres
def func_fitted2(x):
    return (x*b_fit2**2)/((c_fit2+x)**2+(d_fit2+39.7887)**2)*1000
x = np.linspace(10**0,10**3,1000)

#on trace la fonction avec condensateur mais sans la reactance ajustée
#def func_fitted3(x):
#    return (x*b_fit2**2)/(((c_fit2+x)**2)+(-0.001**2))*1000

#print(b_fit1, c_fit1, b_fit2, c_fit2, d_fit2)

fig, ax = plt.subplots()
plt.xscale("log")
ax.plot(x, func_fitted1(x), 'm-', label='Courbe théorique sans le condensateur')
ax.plot(x, func_fitted2(x), 'g-', label='Courbe théorique avec le condensateur et la réactance ajustée')
#ax.plot(x, func_fitted3(x), 'b-', label='Courbe théorique avec le condensateur')
ax.plot(res_1, puiss1, linestyle='none', marker='o', label="Données sans le condensateur")
ax.plot(res_2, puiss2, linestyle='none', marker='o', label="Données avec le condensateur")
ax.set_yticks((0.5, 1, 1.5, 2, 2.5, 3))
ax.set_xticks((10**0,10**1,10**2,10**3))
#plt.title("Intensité en niveaux de gris du trait (en pixels) reliant les deux marqueurs du bas du fantôme. Image 2D")
plt.xlabel("Résistance [Ohm]")
plt.ylabel("Puissance dissipée [mW]")
plt.legend()
plt.show()   