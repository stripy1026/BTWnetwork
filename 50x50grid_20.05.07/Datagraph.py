import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from tqdm import tqdm
import pandas as pd


df1 = pd.read_csv('data_Boundary fixed.csv')
df2 = pd.read_csv('data_f=0.00264.csv')
df3 = pd.read_csv('data_f=0.0015.csv')
df4 = pd.read_csv('data_f=0.004.csv')

#--------------------------

def timetoG( df, probran ):
    t = []
    for i in range( len( df['GC'] ) ):
        t.append( i+1 )

    plt.title('Total grains of 50x50 lattice in BTW model')
    plt.xlabel('time (t)')
    plt.ylabel('G(t)')
    plt.plot( t, df['GC'], 'o', label='f = '+str( probran ) )

    print( 'f = '+str( probran ) )
    print( 'Remains = '+ str( df['GC'][ len( df['GC'] )-1 ] ) )


def timetoG2( df ):
    t = []
    for i in range( len( df['GC'] ) ):
        t.append( i+1 )

    plt.title('Total grains of 50x50 lattice in BTW model')
    plt.xlabel('time (t)')
    plt.ylabel('G(t)')
    plt.plot( t, df['GC'], 'o', label='Boundary fixed' )

    print( 'Boundary fixed' )
    print( 'Remains = '+ str( df['GC'][ len( df['GC'] )-1 ] ) )


def losttosize( df, probran ):
    xl = df['AC']
    yl = [0]
    for i in range( len( df['GC'] ) ):
        if not i == 0:
            y = df['GC'][i] - df['GC'][i-1] -1
            yl.append( -y )

    plt.title('Grain lost in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('L(s)')
    plt.plot( xl, yl, 'o', 'f = '+str( probran ) )


def losttosize2( df ):
    xl = df['AC']
    yl = [0]
    for i in range( len( df['GC'] ) ):
        if not i == 0:
            y = df['GC'][i] - df['GC'][i-1] -1
            yl.append( -y )

    plt.title('Grain lost in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('L(s)')
    plt.plot( xl, yl, 'o', label='Boundary fixed' )


def fractiontosize( df, probran ):
    Mlist = []
    mp = 0
    for i in range( len( df['MC'] ) ):
        mp += df['MC'][i]
        Mlist.append( mp )

    yl = [0]
    for i in range( len( df['GC'] ) ):
        if not i == 0:
            y = df['GC'][i] - df['GC'][i-1] -1
            yl.append( -y )
    Ylist = []
    yp = 0
    for i in range( len( yl ) ):
        yp += yl[i]
        Ylist.append( yp )

    flist = []
    for i in range( len( df['MC'] ) ):
        if Mlist[i] == 0:
            flist.append( 0 )
        else:
            f = Ylist[i]/Mlist[i]
            flist.append( f )

    return flist


def fractiontosize2( df ):
    Mlist = []
    mp = 0
    for i in range( len( df['MC'] ) ):
        mp += df['MC'][i]
        Mlist.append( mp )

    yl = [0]
    for i in range( len( df['GC'] ) ):
        if not i == 0:
            y = df['GC'][i] - df['GC'][i-1] -1
            yl.append( -y )
    Ylist = []
    yp = 0
    for i in range( len( yl ) ):
        yp += yl[i]
        Ylist.append( yp )

    flist = []
    for i in range( len( df['MC'] ) ):
        if Mlist[i] == 0:
            flist.append( 0 )
        else:
            f = Ylist[i]/Mlist[i]
            flist.append( f )

    return flist


def FTSgraph( df, probran ):
    flist = fractiontosize( df, probran )
    plt.title('f in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('f(s)')
    plt.plot( df['AC'][10000:], flist[10000:], 'o', label='f = '+str( probran ) )


def FTSgraph2( df ):
    flist = fractiontosize2( df )
    plt.title('f in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('f(s)')
    plt.plot( df['AC'][10000:], flist[10000:], 'o', label='Boundary fixed' )



def FTSbin( df, probran ):
    flist = fractiontosize( df, probran )
    bin_means, bin_edges, binnumber = binned_statistic( df['AC'][10000:], flist[10000:], statistic='mean', bins=13 )
    plt.title('f in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('f(s)')
    plt.plot( bin_edges[1:], bin_means, 'x-', label='f = '+str( probran ) )


def FTSbin2( df ):
    flist = fractiontosize2( df )
    bin_means, bin_edges, binnumber = binned_statistic( df['AC'][10000:], flist[10000:], statistic='mean', bins=13 )
    plt.title('f in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('f(s)')
    plt.plot( bin_edges[1:], bin_means, 'x-', label='Boundary fixed' )


plt.figure()
timetoG2( df1 )
timetoG( df2, 0.00264 )
#timetoG( df3, 0.0015 )
#timetoG( df4, 0.004 )
plt.legend()
plt.show()

plt.figure()
losttosize2( df1 )
losttosize( df2, 0.00264 )
#losttosize( df3, 0.0015 )
#losttosize( df4, 0.004 )
plt.legend()
plt.show()

plt.figure()
FTSgraph2( df1 )
FTSgraph( df2, 0.00264 )
#FTSgraph( df3, 0.0015 )
#FTSgraph( df4, 0.004 )
plt.legend()
plt.show()

plt.figure()
FTSbin2( df1 )
FTSbin( df2, 0.00264 )
#FTSbin( df3, 0.0015 )
#FTSbin( df4, 0.004 )
plt.legend()
plt.show()
