import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from tqdm import tqdm
import pandas as pd


df1 = pd.read_csv('bin_data_Boundary fixed.csv')
df2 = pd.read_csv('bin_data_f=0.176.csv')
df3 = pd.read_csv('bin_data_f=0.2.csv')
df4 = pd.read_csv('bin_data_f=0.1.csv')


df1e = df1.fillna( method = 'ffill' )
df2e = df2.fillna( method = 'ffill' )
df3e = df3.fillna( method = 'ffill' )
df4e = df4.fillna( method = 'ffill' )
#--------------------------


def Diff( df, probran ):
    dxlist = []
    for i in range( len(df['A'])-1 ):
        dx = df['A'][i+1]
        dxlist.append( dx )

    dylist = []
    for i in range( len(df['B'])-1 ):
        dy = df['B'][i+1] - df['B'][i]
        dylist.append( -dy )


    plt.title('Differential graph of a')
    plt.xlabel('x')
    plt.ylabel('dP( s )')
    plt.plot( dxlist, dylist, 'x-', label='f = '+str( probran ) )


    Yp = 0
    for i in range(5):
        Yp += df['B'][i]
    a = Yp/5
    print( 'a = ', a )



def Diff2( df ):
    dxlist = []
    for i in range( len(df['A'])-1 ):
        dx = df['A'][i+1]
        dxlist.append( dx )

    dylist = []
    for i in range( len(df['B'])-1 ):
        dy = df['B'][i+1] - df['B'][i]
        dylist.append( -dy )


    plt.title('Differential graph of a')
    plt.xlabel('x')
    plt.ylabel('dP( s )')
    plt.plot( dxlist, dylist, 'x-', label='Boundary fixed' )


    Yp = 0
    for i in range(5):
        Yp += df['B'][i]
    a = Yp/5
    print( 'a = ', a )

plt.figure()
Diff2( df1e )
Diff( df2e, 0.176 )
Diff( df3e, 0.2 )
Diff( df4e, 0.1 )
plt.legend()
plt.show()
