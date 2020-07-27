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


df1 = pd.read_csv('data_f=0.0001.csv')
df2 = pd.read_csv('data_f=0.0005.csv')
df3 = pd.read_csv('data_f=0.001.csv')
#df4 = pd.read_csv('data_f=0.00263978.csv')


#--------------------------

def timetoG( df, probran ):
    t = []
    for i in range( len( df['GC'] ) ):
        t.append( i+1 )

    plt.title('Total grains of SF network in BTW model')
    plt.xlabel('time (t)')
    plt.ylabel('G(t)')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot( t, df['GC'], 'o', label='f = '+str( probran ) )

    print( 'f = '+str( probran ) )
    print( 'Remains = '+ str( df['GC'][ len( df['GC'] )-1 ] ) )


def timetoG2( df ):
    t = []
    for i in range( len( df['GC'] ) ):
        t.append( i+1 )

    plt.title('Total grains of SF network in BTW model')
    plt.xlabel('time (t)')
    plt.ylabel('G(t)')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot( t, df['GC'], 'o', label='Boundary fixed' )

    print( 'Boundary fixed' )
    print( 'Remains = '+ str( df['GC'][ len( df['GC'] )-1 ] ) )


def losttosize( df, probran ):
    xl = df['AC']
    yl = [0]
    for i in tqdm( range( len( df['GC'] ) ), desc = 'Drawing graph...' ):
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
    for i in tqdm( range( len( df['GC'] ) ), desc = 'Drawing graph...' ):
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
    for i in tqdm( range( len( df['MC'] ) ), desc = 'Drawing...' ):
        mp += df['MC'][i]
        Mlist.append( mp )

    yl = [0]
    for i in tqdm( range( len( df['GC'] ) ), desc = 'Drawing...'):
        if not i == 0:
            y = df['GC'][i] - df['GC'][i-1] -1
            yl.append( -y )
    Ylist = []
    yp = 0
    for i in tqdm( range( len( yl ) ), desc = 'Drawing...'):
        yp += yl[i]
        Ylist.append( yp )

    flist = []
    for i in tqdm( range( len( df['MC'] ) ), desc = 'Drawing graph...'):
        if Mlist[i] == 0:
            flist.append( 0 )
        else:
            f = Ylist[i]/Mlist[i]
            flist.append( f )

    return flist


def fractiontosize2( df ):
    Mlist = []
    mp = 0
    for i in tqdm( range( len( df['MC'] ) ), desc = 'Drawing...'):
        mp += df['MC'][i]
        Mlist.append( mp )

    yl = [0]
    for i in tqdm( range( len( df['GC'] ) ), desc = 'Drawing...'):
        if not i == 0:
            y = df['GC'][i] - df['GC'][i-1] -1
            yl.append( -y )
    Ylist = []
    yp = 0
    for i in tqdm( range( len( yl ) ), desc = 'Drawing...'):
        yp += yl[i]
        Ylist.append( yp )

    flist = []
    for i in tqdm( range( len( df['MC'] ) ), desc = 'Drawing graph...'):
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


def FTSlogbin( df, probran ):
    flist = fractiontosize( df, probran )
    logx = makelog10( df['AC'][10000:] )
    logy = makelog10( flist[10000:] )
    nplogx = np.array( logx )
    nplogy = np.array( logy )
    bin_means, bin_edges, binnumber = binned_statistic( nplogx, flist[10000:], statistic='mean', bins=20 )
    plt.title('f in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('f(s)')
    plt.plot( bin_edges[1:], bin_means, 'x-', label='f = '+str( probran ) )


def FTSlogbin2( df ):
    flist = fractiontosize2( df )
    logx = makelog10( df['AC'][10000:] )
    logy = makelog10( flist[10000:] )
    nplogx = np.array( logx )
    nplogy = np.array( logy )
    bin_means, bin_edges, binnumber = binned_statistic( nplogx, flist[10000:], statistic='mean', bins=20 )
    plt.title('f in certain size of avalanche')
    plt.xlabel('size (s)')
    plt.ylabel('f(s)')
    plt.plot( bin_edges[1:], bin_means, 'x-', label='Boundary fixed' )




def makelog10( list ):
    loglist = []
    for i, k in enumerate( list ):
        if k == 0:
            loglist.append( 0 )
        else:
            loglist.append( math.log10( k ) )
    return loglist




plt.figure()
timetoG( df1, 0.0001 )
timetoG( df2, 0.0005 )
timetoG( df3, 0.001 )
#timetoG( df4, 0.004 )
plt.legend()
plt.show()

plt.figure()
losttosize( df1, 0.0001 )
losttosize( df2, 0.0005 )
losttosize( df3, 0.001 )
#losttosize( df4, 0.00263978 )
plt.legend()
plt.show()

plt.figure()
FTSgraph( df1, 0.0001 )
FTSgraph( df2, 0.0005 )
FTSgraph( df3, 0.001 )
#FTSgraph( df4, 0.00263978 )
plt.legend()
plt.show()

plt.figure()
FTSbin( df1, 0.0001 )
FTSbin( df2, 0.0005 )
FTSbin( df3, 0.001 )
#FTSbin( df4, 0.00263978 )
plt.legend()
plt.show()

plt.figure()
FTSlogbin( df1, 0.0001 )
FTSlogbin( df2, 0.0005 )
FTSlogbin( df3, 0.001 )
#FTSlogbin( df4, 0.00263978 )
plt.legend()
plt.show()
