import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from tqdm import tqdm
import pandas as pd
from pyvis.network import Network


dd = pd.read_csv( 'DDdata.csv' )

df1 = pd.read_csv('bin_data_f=0.0001.csv')
df2 = pd.read_csv('bin_data_f=0.0005.csv')
df3 = pd.read_csv('bin_data_f=0.001.csv')
#df4 = pd.read_csv('bin_data_f=0.004.csv')


df1e = df1.fillna( method = 'ffill' )
df2e = df2.fillna( method = 'ffill' )
df3e = df3.fillna( method = 'ffill' )
#df4e = df4.fillna( method = 'ffill' )
#--------------------------

def DDplot( data ):
    plt.figure()
    plt.plot(data['A'],data['B'], 'o')
    plt.title('Degree Distribution of SF network BTW model')
    plt.xlabel('Degree k')
    plt.ylabel('P(k)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    logx = makelog10( data['A'] )
    logy = makelog10( data['B'] )
    nplogx = np.array( logx )
    nplogy = np.array( logy )
    popt, pcov = curve_fit( linear_func, nplogx[:len(nplogx)//2], nplogy[:len(nplogy)//2])
    print( 'a = ', popt[:1] )

def makelog10( list ):
    loglist = []
    for i, k in enumerate( list ):
        if k == 0:
            loglist.append( 0 )
        else:
            loglist.append( math.log10( k ) )
    return loglist


def linear_func(x, a, b):
    return a*x + b


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
    plt.title('Avalanche size Distribution of SF network BTW model (log-log)')
    plt.xlabel('Avalanche size( s )')
    plt.ylabel('P( s )')
    plt.plot( df['A'], df['B'], 'x-', label='f = '+str( probran ) )


def bingraph2( df ):
    plt.title('Avalanche size Distribution of SF network BTW model (log-log)')
    plt.xlabel('Avalanche size( s )')
    plt.ylabel('P( s )')
    plt.plot( df['A'], df['B'], 'x-', label='Boundary fixed' )



DDplot( dd )

plt.figure()
Diff( df1e, 0.0001 )
Diff( df2e, 0.0005 )
Diff( df3e, 0.001 )
#Diff( df4e, 0.004 )
plt.legend()
plt.show()

plt.figure()
bingraph( df1e, 0.0001 )
bingraph( df2e, 0.0005 )
bingraph( df3e, 0.001 )
#bingraph( df4e, 0.004 )
plt.legend()
plt.show()
