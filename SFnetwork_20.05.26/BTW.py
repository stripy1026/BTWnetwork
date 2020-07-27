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


#G = nx.grid_2d_graph(5, 5)  # 5x5 grid
#G = nx.random_powerlaw_tree(2500, gamma=3, seed=None, tries=100000) # SF network
G2 = nx.scale_free_graph(2500)
G = G2.to_undirected()
#G = nx.grid_2d_graph(50, 50)  # 50x50 grid

VG = Network( height="750px", width="100%", bgcolor="#222222", font_color="white" )
VG.from_nx( G )
#VG.enable_physics(True)
VG.show_buttons(filter_=['physics'])
VG.show( "mygraph.html" )

# print the adjacency list
#for line in nx.generate_adjlist(G):
#    print(line)
# write edgelist to grid.edgelist
nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
# read edgelist from grid.edgelist
H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

#nx.draw(H)
#plt.show()

degrees = dict(G.degree())
degree_values = sorted(set(degrees.values()))
histogram = [list(degrees.values()).count(i)/float(nx.number_of_nodes(G)) for i in degree_values]

Ddf = pd.DataFrame( { 'A': degree_values,
                    'B': histogram } )
Ddf.to_csv('DDdata.csv')

plt.figure()
plt.plot(degree_values,histogram, 'o')
plt.title('Degree Distribution of SF network BTW model')
plt.xlabel('Degree k')
plt.ylabel('P(k)')
plt.xscale('log')
plt.yscale('log')
plt.show()

SB5 = { (0,0) : 0, (0,1) : 0, (0,2) : 0, (0,3) : 0, (0,4) : 0,
       (1,0) : 0, (1,1) : 0, (1,2) : 0, (1,3) : 0, (1,4) : 0,
       (2,0) : 0, (2,1) : 0, (2,2) : 0, (2,3) : 0, (2,4) : 0,
       (3,0) : 0, (3,1) : 0, (3,2) : 0, (3,3) : 0, (3,4) : 0,
       (4,0) : 0, (4,1) : 0, (4,2) : 0, (4,3) : 0, (4,4) : 0 }



SFkey = []
SFval = []
for i in range(2500):
    SFkey.append( i )
    SFval.append( 0 )

SFN = dict( zip(SFkey, SFval) )



key = []
for i in range(50):
    for j in range(50):
        key.append( (i,j) )
#print( key )
value = []
for i in range(2500):
    value.append( 0 )
#print( value )

SB50 = dict( zip(key, value) )




#nx.set_node_attributes(G, SFN, 'sand')
#print( G.nodes[0]['sand'] )
#print( G.nodes[14]['sand'] )
#print( G.nodes[99]['sand'] )

def avalanche( grid, probran=0.1 ):
    ac = 0
    mc = 0
    while True:
        changed = False
#        if ac > 0:
#            sink( grid, probran )
        for i, key in enumerate( grid ):
            if grid[key] >= len( list( G[key].keys() ) ): #1
                grid[key] -= len( list( G[key].keys() ) ) #1
#2            if grid[key] >= 4: #2
#2                grid[key] -= 4 #2
                for i, key in enumerate( G[key] ):
                    mc += 1
#                    print( "Neigbors = ", key, " : ", G.nodes[key]['sand'] )
#                    G.nodes[key]['sand'] += 1
#                    print( "Neigbors after = ", key, " : ", G.nodes[key]['sand'] )
                    s = random.random()
                    if s >= probran:
                        grid[key] += 1
                ac += 1
                changed = True
        if not changed:
            break
    return grid, ac, mc, changed


def avalanche2( grid ):
    ac = 0
    mc = 0
    while True:
        changed = False
#        if ac > 0:
#            sink( grid, probran )
        for i, key in enumerate( grid ):
#1            if grid[key] >= len( list( G[key].keys() ) ): #1
#1                grid[key] -= len( list( G[key].keys() ) ) #1
            if grid[key] >= 4: #2
                grid[key] -= 4 #2
                for i, key in enumerate( G[key] ):
                    mc += 1
#                    print( "Neigbors = ", key, " : ", G.nodes[key]['sand'] )
#                    G.nodes[key]['sand'] += 1
#                    print( "Neigbors after = ", key, " : ", G.nodes[key]['sand'] )
                    grid[key] += 1
                ac += 1
                changed = True
        if not changed:
            break
    return grid, ac, mc, changed



def simulate( grid, t=100000, probran=0.1 ):
    AC = []
    MC = []
    TO = []
    for i in tqdm( range( t ), desc="Avalanche process..." ):
        toppling( grid )
        grid, ac, mc, changed = avalanche( grid, probran )
        AC.append( ac )
        MC.append( mc )
        ls = []
        for i, key in enumerate( grid ):
            ls.append( grid[key] )
        total = sum(ls)
        TO.append( total )
    M = sum(MC)
    nx.set_node_attributes(G, grid, 'sand')
    grid, remains = showkey( grid )
    lost = (t-remains)/M
    print( 'f = ', lost )
#    ASDplot( AC, probran )
    binplot( AC, probran )

    df2 = pd.DataFrame( { 'AC': AC,
                    'MC': MC,
                    'GC': TO } )
    df2.to_csv('data_f='+str( probran )+'.csv')

    return grid, AC


def simulate2( grid, t=100000 ):
    AC = []
    MC = []
    TO = []
    for i in tqdm( range( t ), desc="Avalanche process..." ):
        toppling( grid )
        grid, ac, mc, changed = avalanche2( grid )
        AC.append( ac )
        MC.append( mc )
        ls = []
        for i, key in enumerate( grid ):
            ls.append( grid[key] )
        total = sum(ls)
        TO.append( total )
    M = sum(MC)
    nx.set_node_attributes(G, grid, 'sand')
    grid, remains = showkey( grid )
    lost = (t-remains)/M
    print( 'f = ', lost )
#    ASDplot( AC )
    binplot2( AC )

    df2 = pd.DataFrame( { 'AC': AC,
                    'MC': MC,
                    'GC': TO } )
    df2.to_csv('data_Boundary fixed.csv')

    return grid, AC



def pre_simulate( grid, t=100, probran=0.1 ):
    AC = []
    MC = []
    for i in tqdm( range( t ), desc="pre-simulating..." ):
        toppling( grid )
        grid, ac, mc, changed = avalanche2( grid, probran )
        AC.append( ac )
        MC.append( mc )
    nx.set_node_attributes(G, grid, 'sand')
#    ASDplot( AC )
#    binplot( AC )
    return grid, AC



def showkey( grid ):
    ls = []
    for i, key in enumerate( grid ):
        ls.append( grid[key] )
#    print( ls )
    remains = sum(ls)
    print( 'The number of remains = ', sum(ls) )
    print( 'Remains per node = ', sum(ls)/2500 )
    return grid, remains



def sink( grid, probran=0.1 ):
    for i, key in enumerate( grid ):
        s = random.random()
        if s < probran:
            if grid[key] > 0:
                grid[key] -= 1



def toppling( grid ):
    i = random.randrange(0,2500)
    grid[i] += 1




def ASDplot( ls, probran ):
    lsd = dict( zip( range( len(ls) ), ls ) )
    lsd_values = sorted(set(lsd.values()))
    lsdhist = [list(lsd.values()).count(i)/float( len(ls) ) for i in lsd_values]
    plt.plot(lsd_values,lsdhist, 'o', label='f = '+str( probran ) )
    plt.title('Avalanche size Distribution of SF network BTW model')
    plt.xlabel('Avalanche size s')
    plt.ylabel('P(s)')
    plt.xscale('log')
    plt.yscale('log')




def binplot( ls, probran ):
    lsd = dict( zip( range( len(ls) ), ls ) )
    lsd_values = sorted(set(lsd.values()))
    lsdhist = [list(lsd.values()).count(i)/float( len(ls) ) for i in lsd_values]
#    print( 'x = ', lsd_values )
#    print()
#    print( 'y = ', lsdhist )
    logx = [0] + makelog10( lsd_values )
    logy = [ math.log10( lsdhist[0] ) ] + makelog10( lsdhist )
#    print( 'logx = ', logx)
    nplogx = np.array( logx )
    nplogy = np.array( logy )
#    print( 'nx = ', nplogx, 'lenght = ', len(nplogx) )
#    print()
#    print( 'ny = ', nplogy, 'lenght = ', len(nplogy) )
    popt, pcov = curve_fit( linear_func, nplogx[:len(nplogx)//2], nplogy[:len(nplogy)//2])
    print( 'a = ', popt[:1] )


    bin_means, bin_edges, binnumber = binned_statistic( nplogx[1:], nplogy[1:], statistic='mean', bins=10 )
#    plt.figure()
#    plt.plot( nplogx[1:], nplogy[1:], 'o' )
    plt.title('Avalanche size Distribution of SF network BTW model (log-log)')
    plt.xlabel('Avalanche size( s )')
    plt.ylabel('P( s )')
#    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5)
    plt.plot( bin_edges[1:], bin_means, 'x-', label='f = '+str( probran ) )
#    plt.show()

#    store = open('bin_data_f='+str( probran )+'.txt', 'w')
#    print( bin_edges[1:], bin_means, file=store )
#    store.close()

    df = pd.DataFrame( { 'A': bin_edges[1:],
                    'B': bin_means } )
    df.to_csv('bin_data_f='+str( probran )+'.csv')


def binplot2( ls ):
    lsd = dict( zip( range( len(ls) ), ls ) )
    lsd_values = sorted(set(lsd.values()))
    lsdhist = [list(lsd.values()).count(i)/float( len(ls) ) for i in lsd_values]
#    print( 'x = ', lsd_values )
#    print()
#    print( 'y = ', lsdhist )
    logx = [0] + makelog10( lsd_values )
    logy = [ math.log10( lsdhist[0] ) ] + makelog10( lsdhist )
#    print( 'logx = ', logx)
    nplogx = np.array( logx )
    nplogy = np.array( logy )
#    print( 'nx = ', nplogx, 'lenght = ', len(nplogx) )
#    print()
#    print( 'ny = ', nplogy, 'lenght = ', len(nplogy) )
    popt, pcov = curve_fit( linear_func, nplogx[:len(nplogx)//2], nplogy[:len(nplogy)//2])
    print( 'a = ', popt[:1] )


    bin_means, bin_edges, binnumber = binned_statistic( nplogx[1:], nplogy[1:], statistic='mean', bins=10 )
#    plt.figure()
#    plt.plot( nplogx[1:], nplogy[1:], 'o' )
    plt.title('Avalanche size Distribution of SF network BTW model (log-log)')
    plt.xlabel('Avalanche size( s )')
    plt.ylabel('P( s )')
#    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5)
    plt.plot( bin_edges[1:], bin_means, 'x-', label='Boundary fixed' )
#    plt.show()

#    store = open('bin_data_Boundary fixed.txt', 'w')
#    print( bin_edges[1:], bin_means, file=store )
#    store.close()

    df = pd.DataFrame( { 'A': bin_edges[1:],
                    'B': bin_means } )
    df.to_csv('bin_data_Boundary fixed.csv')



def makelog10( list ):
    loglist = []
    for i, k in enumerate( list ):
        if i > 0:
            loglist.append( math.log10( k ) )
    return loglist
#    print( 'logvalue = ', logx )



def linear_func(x, a, b):
    return a*x + b



###################################################################


nx.set_node_attributes(G, SFN, 'sand')
#print( 'initial state =' )
#showkey( SB5 )

plt.figure()
SFNa = SFN.copy()
simulate( SFNa, t=100000, probran=0.0001 )
SFNb = SFN.copy()
simulate( SFNb, t=100000, probran=0.0005 )
SFNc = SFN.copy()
simulate( SFNc, t=100000, probran=0.0010 )
SFNd = SFN.copy()
simulate( SFNd, t=100000, probran=0.00001 )
plt.legend()
plt.show()




# data = np.array( )



#print( G[(0,0)] )
#print( G[(1,2)] )
#print( G[(4,3)] )
#print( [ n for n in G[(1,2)] ] )