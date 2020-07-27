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
df2 = pd.read_csv('bin_data_f=0.00264.csv')
df3 = pd.read_csv('bin_data_f=0.0015.csv')
df4 = pd.read_csv('bin_data_f=0.004.csv')


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
        dx = df['A'][i+1] - df['A'][i]
        dy = df['B'][i+1] - df['B'][i]
        dylist.append( -dy/dx )


    plt.title('Differential graph of a')
    plt.xlabel('x')
    plt.ylabel('dP( s )')
    plt.plot( dxlist, dylist, 'x-', label='f = '+str( probran ) )


    Yp = 0
    for i in range(6):
        Yp += dylist[i]
    a = Yp/6
    print( 'a = ', a )



def Diff2( df ):
    dxlist = []
    for i in range( len(df['A'])-1 ):
        dx = df['A'][i+1]
        dxlist.append( dx )

    dylist = []
    for i in range( len(df['B'])-1 ):
        dx = df['A'][i+1] - df['A'][i]
        dy = df['B'][i+1] - df['B'][i]
        dylist.append( -dy/dx )


    plt.title('Differential graph of a')
    plt.xlabel('x')
    plt.ylabel('dP( s )')
    plt.plot( dxlist, dylist, 'x-', label='Boundary fixed' )


    Yp = 0
    for i in range(6):
        Yp += dylist[i]
    a = Yp/6
    print( 'a = ', a )


def bingraph( df, probran ):
    plt.title('Avalanche size Distribution of 2D 50x50 lattice BTW model (log-log)')
    plt.xlabel('Avalanche size( s )')
    plt.ylabel('P( s )')
    plt.plot( df['A'], df['B'], 'x-', label='f = '+str( probran ) )


def bingraph2( df ):
    plt.title('Avalanche size Distribution of 2D 50x50 lattice BTW model (log-log)')
    plt.xlabel('Avalanche size( s )')
    plt.ylabel('P( s )')
    plt.plot( df['A'], df['B'], 'x-', label='Boundary fixed' )



plt.figure()
Diff2( df1e )
Diff( df2e, 0.0026418437251063704 )
Diff( df3e, 0.0015 )
Diff( df4e, 0.004 )
plt.legend()
plt.show()

plt.figure()
bingraph2( df1e )
bingraph( df2e, 0.0026418437251063704 )
bingraph( df3e, 0.0015 )
bingraph( df4e, 0.004 )
plt.legend()
plt.show()
