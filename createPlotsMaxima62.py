import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator#, FormatStrFormatter
import itertools
import os
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import copy 
import csv

plt.ioff()

xs, xt , ys, yt = {}, {}, {}, {}
gfd, gfx, gfy, gl = {}, {}, {}, {} 
dtp = {}   # data to plot, using filename as the key
l = {}  # l[filename]  is a labels/maxima for the file
yTop, yBot, yTopTrim, yBotTrim ={}, {}, {}, {} 
ref = [] #reference data file names 
atp ={}
 
numberofPlotsDrawnLastTime = 30
#number of units the lines within each plot should be separated by
stagger = 3     
staggersq = -2

#amount of extra border above and below graphs
yBorder = 1.0  
                   
##setup hashmaps for y axis limits
yLow = {}
yHigh = {}
graphs = {}
aptgraphs = {}

# group level plot label -- find maxima for a given compound:
grplabels = {}
grpfoundDiff = {}
grpfoundX = {}
grpfoundY = {}

#set global var for amt the found maxima can differ from
#theoretcial maxima  
maximaTolerance = 0.1

#distance--how far upwards the label for each maximum should be moved
labelDistance = 0.2

#set global variables for minor axis ticks
reg_minorLocator = MultipleLocator(.1) 
stag_minorLocator = MultipleLocator(1) 

#set global var for which sub graphs should be created:
sol = ''
conc = ''

mixes = {'1': 'NaOH-slag',
 '10': 'Na2CO3-slag',
 '11': 'Na2CO3-metakaolin',
 '12': 'Na2CO3-50% slag, 50% metakaolin',
 '13': 'Na2SiO3-slag',
 '14': 'Na2SiO3-metakaolin',
 '15': 'Na2SiO3-50% slag, 50% metakaolin',
 '16': 'K2CO3-slag',
 '17': 'K2CO3-metakaolin',
 '2': 'NaOH-metakaolin',
 '3': 'NaOH-50% slag, 50% metakaolin',
 '4': 'KOH-slag',
 '5': 'KOH-metakaolin',
 '6': 'KOH-50% slag, 50% metakaolin',
 '7': 'Na2SO4-slag',
 '8': 'Na2SO4-metakaolin',
 '9': 'Na2SO4-50% slag, 50% metakaolin'}


        
                  
def getStartLine(group):
    '''sets the line in which the data to be plotted starts in the file
    to ignore the rest of the preceding lines in the file'''
    
    if group == 'G16':
        startingLine = 144
    else:
        startingLine = 142
        
    return startingLine




def setAxesTitles_Ax(axName, plotType):
    '''for ax object:
    sets the labels for the x and y axes for gr and sq graphs
    uses matplotlib built in called mathtext, similar to LaTeX'''
    
    if plotType == 'gr':
        axName.set_xlabel(r'$r (\AA)$'), axName.set_ylabel(r'$G(r) (\AA^{-2})$')
    elif plotType == 'sq':
        axName.set_xlabel(r'$ Q (\AA^{-1}) $'), axName.set_ylabel(r'$S(Q)$')
        

def setAxesTitles_plt(plotType, fontsize=None):
    '''using pyplot (plt) (not ax object):
    sets the labels for the x and y axes for gr and sq graphs
    uses matplotlib built in called mathtext, similar to LaTeX'''
    
    if plotType == 'gr':
        plt.xlabel(r'$r (\AA)$', fontsize=fontsize), plt.ylabel(r'$G(r) (\AA^{-2})$', fontsize=fontsize)
    elif plotType == 'sq':
        plt.xlabel(r'$ Q (\AA^{-1}) $', fontsize=fontsize), plt.ylabel(r'$S(Q)$', fontsize=fontsize)        
            
                                    
def savePlots(fig, plotName, group, path):
    '''saves the plots as different file types for later use'''
   
    os.chdir(path)
    plt.figure(plotName)
    plt.savefig(plotName + group + '.png', format='png')
    plt.savefig(plotName+ group +'.ps', format='ps')
    plt.savefig(plotName+group +'.svg', format='svg')
    plt.savefig(plotName+group +'.pdf', format='pdf')
    plt.savefig(plotName+group +'.eps', format='eps')
    
    os.chdir('C:\Maya\LLP\png2')
    plt.savefig(plotName + ' ' + group +'.png', format='png')


def findMaxima(dataFile, group,  grpfoundDiff, grpfoundX, grpfoundY, maximaTolerance = 0.10 ):
    '''returns dictionary with all the coordinates and labels for all the peaks
    in a particular graph
    maxima tolerance is amount the maximum can differ from the maximum in the paper'''
    
    #maximaTolerance = 0.10
  
    
    #function taken from this website:
    #http://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficien 
    def detect_local_maxima(arr):
        # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        Takes an array and detects the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        Modified from detect_local_minima
        """
        # define an connected neighborhood
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
        neighborhood = morphology.generate_binary_structure(len(arr.shape),2)
        # apply the local  filter; all locations of minimum value 
        # in their neighborhood are set to 1
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
        local_min = (filters.maximum_filter(arr, footprint=neighborhood)==arr)
        # local_min is a mask that contains the peaks we are 
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.
        # 
        # we create the mask of the background
        background = (arr==0)
        # 
        # a little technicality: we must erode the background in order to 
        # successfully subtract it from local_min, otherwise a line will 
        # appear along the background border (artifact of the local minimum filter)
        # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
        eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
        # 
        # we obtain the final mask, containing only peaks, 
        # by removing the background from the local_min mask
        detected_minima = local_min - eroded_background
        return np.where(detected_minima)    
        
    #reads data from file, starting from the line in the file on which the
    #data begins
    with open(dataFile, 'r') as file:
        curr_lines = list(itertools.islice(file, getStartLine(group), None))  
   # file.close()    
    #example of the above:    
    #with open('1-1 NaOH-slag Na2SO4-1%-20degC smt.gr', 'r') as file:
    #    curr_lines = list(itertools.islice(file, 251, 268))
    
                
    #creates lists of points to be plotted       
    xSeries = []
    ySeries = []
    
    #gets data from file
    for line in curr_lines:   
        values = line.split('   ')
        if 1 <= float(values[0]) <=6:
            xSeries.append(values[0])
            ySeries.append(values[1])
    
    #formats the points correctly:
    length = len(xSeries)                   
    for series in [xSeries, ySeries]:
        for i in range(length):#makes sure both x and y data sets are the same length    
            series[i]=series[i].strip()#removes any white space for each point#
            series[i]=float(series[i])#makes each point a float 
    
    
    #creates zero numpy array for coordinates    
    a = np.zeros(shape=(length,2))   
                                            
    #assigns the xy coordinates to the numpy array
    for i in range(length):
        a[i,0], a[i,1]=xSeries[i], ySeries[i]
    
    
    #finds the local maxima of the y coordinates:
    
    #converts to np array then makes a new array with the local maxima
    #of the array
    ym=np.array(ySeries)
    ymax=detect_local_maxima(ym)    
    #print 'ymax ', ymax
    
    #stores number of maxima
    lenYMax= ymax[0].size
    #print '\n', 'lenYmax', lenYMax
    
    #stores all the maxima in numpy array
    maxima = np.zeros(shape=(lenYMax,2))
    for i in range(lenYMax):
        index = ymax[0][i]
        maxima[i,0], maxima[i,1] = xSeries[index], ySeries[index]
    
    print '\n', 'maxima', maxima, '\n'
    
    
    #determine if the maxima found above correspond to the ones in the papers\
    
    #dictionary of maxima from the papers
    acceptedMaxima = {
    1.64: 'Si-O',
    2.02: 'Mg-O \n Al-O', \
    #2.3: 'Na-O', \
    2.35: 'Ca-O', \
    2.7: 'O-O',
    3.1: 'Si-Si \n Si-Al',
    3.63: 'Ca-Si'
    }
    
       
    #for i in acceptedMaxima.iterkeys():
    #    print i
    #
    #print '\n'
    #
    #for num in range(lenYMax):
    #    print maxima[num][0]
    
    #print '\n'
    
    labels = {}
    #foundDiff holds the x axis distance from the accepted value to find the max
    #that is closest to the accepted value 
        #example:
        #accepted value for Si=O: 1.64, experimental x value = 1.68, diff =.04
        #next experimental value: 1.61, diff = .03
        #1.61 will become the new x coordinate to be labeled
    foundDiff = {}
    foundX = {}
    foundY = {}
    # grpfoundDiff, grpfoundX, grpfoundY = {}, {}, {}

    #if the experimental maxima are within maxima_tolerance of the maxima in the 
    #papers, create labels for these maxima
    for i in acceptedMaxima.iterkeys():
        for num in range(lenYMax):
            diff = abs(i - maxima[num][0])
            if diff <= maximaTolerance:
                #print 'i', i
                #print 'max', maxima[num][0]
                #print 'diff', i - maxima[num][0]
                #print 'abs', abs(i - maxima[num][0])
                #if maxima[num][1] < 0:
                #    maxima[num][1] = .1
                compound = acceptedMaxima[i]
                xValue = maxima[num][0]
                yValue = maxima[num][1]
                #if the current difference from accepted value is lower than 
                #previously found lowest difference, replace old lowest with new lowest
                if compound not in foundDiff:
                    foundDiff[compound] = diff 
                    foundX[compound] = xValue
                    foundY[compound] = yValue
                else:
                    if diff < foundDiff[compound]:
                        foundDiff[compound] = diff 
                        foundX[compound] = xValue
                        foundY[compound] = yValue
                        
    if 'O-O' not in foundDiff:
        foundDiff['O-O'] = 0.1
        foundX['O-O']    = 2.6
        foundY['O-O']    =-0.2
    
    print '\nfoundDiff:', foundDiff, '\n'
                
    for compound1 in foundDiff:   
        #make a collection of found
        labels[(round(foundX[compound1], 2), round(foundY[compound1], 2))] = compound1         
               # labels[(round(maxima[num][0], 2), maxima[num][1])] = acceptedMaxima[i] 
        # now make group level determinations       
        if compound1 not in grpfoundDiff:
            grpfoundDiff[compound1] = foundDiff[compound1]
            grpfoundX[compound1] = foundX[compound1]
            grpfoundY[compound1] = foundY[compound1] 
        else:
            if grpfoundDiff[compound1] < foundDiff[compound1]:
                grpfoundDiff[compound1] = foundDiff[compound1]
                grpfoundX[compound1] = foundX[compound1]
                grpfoundY[compound1] = foundY[compound1] 
             
            
    
    print 'labels', labels, '\n'
    #print 'grplabels', grplabels, '\n'
    return labels, grpfoundDiff, grpfoundX, grpfoundY


def labelMaxima(labels, fig, fontsize=None):#, ax):
    '''adds labels to the peaks in a figure'''
    
    #plt.figure(fig)
    for i in labels:#.iterkeys():
    #for i in range(len(labels)):
        print 'i', i,'labels[i] ', labels[i], '\n'
        plt.annotate(labels[i], xy=i, xytext = (i[0], i[1] + labelDistance), fontsize=fontsize)
        
        
        #draw dashed vertical lines from maximum to axis:
        
        #doesnt work
        #ax.add_line(Line2D([i[0],i[0]], [i[1],0], transform=ax.transAxes,
         #         linewidth=2, color='b'))
        
        #works
        #plt.plot([i[0],i[0]], [i[1],-20], linestyle='dashed')
    




def writeMaxima(labels, current_file, path, group, f_new):
    '''creates 1 png image file for each of the current files and uses PIL
    to draw the text containing the maxima on that image
    -also writes the maxima for each file to the 1 text file for the 
    entire group'''
    
    os.chdir('C:\Maya\LLP')
        
    font = ImageFont.truetype("Arial.ttf",14)
    img=Image.new("RGBA", (300,200),(255,255,255))
    draw = ImageDraw.Draw(img)
    
    draw.text((0, 0), current_file[:-3] ,(0,0,0),font=font)
    
    a=17
    
    for key in labels:
        coords = (0, a)
        if '\n' in labels[key]:
            labels[key] = labels[key].replace('\n', '')
        draw.text(coords, str(key) + ': ' + labels[key] ,(0,0,0),font=font)
        a += 17
    
    draw = ImageDraw.Draw(img)
    
    os.chdir('C:\Maya\LLP\png')
    img.save(current_file[:-2] + 'png')
    
    os.chdir(path)
    
    #writes to text file all the maxima and minima
    f_new.write(current_file[:-2] + '\n')
    for key in labels:
        if '\n' in labels[key]:
            labels[key] = labels[key].replace('\n', '')
        f_new.write(str(key) + ': ' + labels[key] + '\n')
    f_new.write('\n')
    
    
    os.chdir('C:\Maya\LLP')    

def plotData(path, plot, file1, regAx, stagAx, group, staggerCount, f_new, grpfoundDiff, grpfoundX, grpfoundY, lastFile):
    '''for each data file, plots that file's data on the figures'''
    
    global yLow, yHigh, aptgraphs   
    global xs, xt, ys, yt
    global gfd, gfx, gfy, gl
    global dtp    # data to plot
    global ref    
    
    os.chdir(path)
    
    #opens the file, ignoring first 141 lines which don't have data
    with open(file1, 'r') as file:
        curr_lines = list(itertools.islice(file, getStartLine(group), None))
    
    print file1
    #print curr_lines       
                
    #creates lists of points to be plotted       
    xSeries = []
    ySeries = []
    
    #gets data from file
    for line in curr_lines:    
        values = line.split('   ')
        xSeries.append(values[0])
        ySeries.append(values[1])
    xSeries[0] = xSeries[0].strip()
    #print xSeries
    #print ySeries
    xSeries = map(float, xSeries)
    ySeries = map(float, ySeries)           


    #plots the current file's data:
 
    #creates lists of points  (trimmmed by xSeries within 1 to 6 --
    #  for regular graphs to be plotted       
    #  for staggered graphs  available x values need to be plotted
    xSeriesTrim = []
    ySeriesTrim = []   
    length = len(xSeries)   
    #makes sure both x and y data sets are the same length
    #removed any white space for each pointf above ^^
    #makes each point a float                      
    for i in range(length):
        if 1.0 <= xSeries[i] <=6.0:
            xSeriesTrim.append(xSeries[i])
            ySeriesTrim.append(ySeries[i])

    xs[file1] = copy.deepcopy(xSeries)  
    ys[file1] = copy.deepcopy(ySeries)
    xt[file1] = copy.deepcopy(xSeriesTrim)
    yt[file1] = copy.deepcopy(ySeriesTrim)
    
    #plots regular 
    plt.figure('regular '+ plot)  
    
    
    #makes ref samples black color and have dashed or dotted line style
    #if not a ref, leaves it set to default
    if 'Ref' in file1:
        if 'air' in file1: #ref air
            regAx.plot(xSeriesTrim, ySeriesTrim, 'k-.', label=file1)
        else: #ref pure water
            regAx.plot(xSeriesTrim, ySeriesTrim, 'k:', label=file1)
    else:        
        regAx.plot(xSeriesTrim, ySeriesTrim , label=file1)
    
    #creates minor axis ticks
    regAx.xaxis.set_minor_locator(reg_minorLocator)
    
    plt.xlim(1, 6)
    
    #for staggered graphs--use entire domain
    upperLim = max(ySeries)
    lowerLim = min(ySeries)
    #for regular graphs--use only x values from 1 to 6
    upperLimTrim = max(ySeriesTrim)
    lowerLimTrim = min(ySeriesTrim)
    
    yTop[file1] = upperLim
    yBot[file1] = lowerLim
    yBotTrim[file1] = lowerLimTrim
    yTopTrim[file1] = upperLimTrim
    
    
    #for sub-graphs that compare only a few of the files
    #aptgraphs creates a list that holds all the graphs that this file should
    #be used for:
    
    #sol and conc are global variables
    global sol, conc
    
    
    aptgraphs = []
    if 'Ref' not in file1: # or 'Scan' not in file1 or '-' not in file1:
        print 'Deriving mapping of graphs from file name', file1
        # example 1: 2-11  H2SO4-1% smt.sq
        # example 2: 2-1  Na2SO4-1% smt.sq
        notUsed1, blank, solConc, tail  = file1.split(' ')
        sol, conc = solConc.split('-')
        notUsed2, graphType = tail.split('.')
        print 'solution, concentration, graphType ', sol, conc, graphType
       
        graphName = 'Regular'   +  ' ' + graphType
    
        aptgraphs.append(graphName)
        if graphName not in dtp:
           dtp[graphName] = []
        if file1 not in dtp[graphName]:
            dtp[graphName].append(file1)

        graphName = 'Staggered' +  ' ' + graphType     
        aptgraphs.append(graphName)   # staggered needs stagggerCount * stagger  added to yLow, yHigh
        if graphName not in dtp:
           dtp[graphName] = []
        if file1 not in dtp[graphName]:
            dtp[graphName].append(file1)        

        graphName = sol +  ' ' + graphType 
        aptgraphs.append(graphName)
        if graphName not in dtp:
           dtp[graphName] = []
        if file1 not in dtp[graphName]:
            dtp[graphName].append(file1)        

            
        if conc == '5%':
            graphName = conc +  ' ' + graphType 
            aptgraphs.append(graphName)
            if graphName not in dtp:
                dtp[graphName] = []
            if file1 not in dtp[graphName]:
                dtp[graphName].append(file1)        

    else: 
        if file1 not in ref:        
            ref.append(file1) #reference will be added to every graph at the
        #time of plotting
#        # for 'Ref' in file1 
#        graphType =  file1[-2:]
        
#        aptgraphs.append('Regular'   +  ' ' + graphType)
#        aptgraphs.append(sol         +  ' ' + graphType)
#        aptgraphs.append(conc     +  ' ' + graphType)
    
#    if graphName not in dtp:
#        dtp[graphName] = []
#    if file1 not in dtp[graphName]:
#        dtp[graphName].append(file1)    
    
    print '\nfile1', file1, '\naptgraphs', aptgraphs
        
                
    for i in aptgraphs:
        if ('Regular' in i) or ('Ref' in file1):
            if upperLimTrim > yHigh.get(i):
                print 'For graph', i, ' Increasing yHigh from ', yHigh.get(i), ' to ',  upperLimTrim  
                yHigh[i] = upperLimTrim
            if lowerLimTrim < yLow.get(i):
                print 'For graph', i, ' Decreasing yLow from ',  yLow.get(i), ' to ', lowerLimTrim  
                yLow[i] = lowerLimTrim
        elif ('Staggered' in i) or ('Ref' in file1):   
            if upperLim > yHigh.get(i):
                print 'For graph', i, ' Increasing yHigh from ', yHigh.get(i), ' to ',  upperLim  
                yHigh[i] = upperLim
            if lowerLim < yLow.get(i):
                print 'For graph', i, ' Decreasing yLow from ',  yLow.get(i), ' to ', lowerLim  
                yLow[i] = lowerLim
                
    #creates legend, sets font size and legend location ,sets title
    plt.legend(prop={'size':7})#, loc=3)
    plt.title(group) 
        
    #fig = plt.gcf()
    #fig.subplots_adjust(bottom=0.2)
    #plt.tight_layout()
    for i in aptgraphs:
        plt.figure(i)
        
        if 'Ref' in file1:
            if 'air' in file1:
                plt.plot(xSeriesTrim, ySeriesTrim, 'k-.', label=file1)
            else:
                plt.plot(xSeriesTrim, ySeriesTrim, 'k:', label=file1)
        else:        
            plt.plot(xSeriesTrim, ySeriesTrim , label=file1)
        
        plt.legend(prop={'size':7})
        setAxesTitles_plt(i[-2:])
        #plt.ylim(yLow, yHigh)
 
    #labels gr graphs: 
    if plot == 'gr':
        labels, grpfoundDiff, grpfoundX, grpfoundY = findMaxima(file1, group, grpfoundDiff, grpfoundX, grpfoundY )
        # next 3 lines print first label maxima - works  - do not remove 
        
        #plot group level maxima
        l[file1] = copy.deepcopy(labels)

        print 'grpfoundDiff'
        print grpfoundDiff
  
    #    upperLim = max(ySeries)
    #    lowerLim = min(ySeries)
    #
    #    if upperLim > yHigh:
    #        print 'Increasing yHigh from ', yHigh, ' to ',  upperLim  
    #        yHigh = upperLim
    #    if lowerLim < yLow:
    #        print 'Decreasing yLow from ',  yLow, ' to ', lowerLim  
    #        yLow = lowerLim   
          
        
        #makes final changes to graph once all the files have been plotted
        print 'lastFile:', lastFile
        if lastFile == True:
            print 'yLow:\n', yLow 
            print 'yHigh:\n', yHigh
            
            grplabels = {}  
            
            for i  in aptgraphs:
                 if ( 'gr' and 'Staggered' not in i ):  # no labels for staggered gr graphs
                #if ( 'gr' not in i and 'Staggered' not in i ): #using this all labels are gone
                    compound = 'O-O'
                    if compound not in grpfoundDiff:
                        print 'no o-o found at group level after last file has been processed'
                        print 'adding new o-o since findMaxima function doesnt detect o-o points on shoulder'
                        grpfoundDiff[compound] = 0.1  
                        grpfoundX[compound] = 2.6
                        grpfoundY[compound] = -0.2                         
                    else:
                        print 'o-o has been found; o-o check not run'
                        
                    print 'Setting up Group Labels:' 
                    grplabels = {}     
        
                            
                    for compound1 in grpfoundDiff:   
                        x = round(grpfoundX[compound1],2)
                        y = round(grpfoundY.get(compound1),2)
                        print ('x, y, compound', (x, y, compound1))  
                        grplabels[(x, y)] = compound1      
                                
                    print '\ngrplabels:\n', grplabels       
                    #labelMaxima(grplabels, 'regular gr',regAx)
                    labelMaxima(grplabels, i) 
     
            # gfd, gfx, gfy, gl 
                    
            gfd = copy.deepcopy(grpfoundDiff)
            gfx = copy.deepcopy(grpfoundX)
            gfy = copy.deepcopy(grpfoundY)
            gl =  copy.deepcopy(grplabels)
                                                                        
           

          
            #for i in aptgraphs:      
            #        if 'Regular' in i:
            #            print '\nIncreasing yHigh value ', yHigh[i], 'by border value', yBorder
            #                    #plt.ylim(yLow-yBorder, yHigh+yBorder)
            #                    
            #            curHigh = yHigh.get(i) + yBorder
            #            curLow = yLow.get(i) - yBorder 
            #            print '\ncurHigh', curHigh, '\ncurLow', curLow
            #            plt.figure(i)
            #            # plt.ylim(curLow, curHigh)

            for i in graphs:      
                    if 'regular gr' in i:
                        print '\nIncreasing yHigh value ', yHigh[i], 'by border value', yBorder
                                #plt.ylim(yLow-yBorder, yHigh+yBorder)
                                
                        curHigh = yHigh[i] + yBorder
                        curLow = yLow[i] - yBorder 
                        print '\ncurHigh', curHigh, '\ncurLow', curLow
                        plt.figure(i)
                        # plt.ylim(curLow, curHigh)


            
        writeMaxima(labels, file1, path, group, f_new)    
    
    #plt.annotate('Si-O', xy=(1.62, 2.1754530000000001))

    #plots staggered 
    plt.figure('staggered ' + plot)  
    
    yStaggered = []
    yStaggered = [pt + staggerCount for pt in ySeries]
    
    if 'Ref' in file1:
        if 'air' in file1:
            stagAx.plot(xSeries, yStaggered, 'k-.', label=file1)
        else:
            stagAx.plot(xSeries, yStaggered, 'k:', label=file1)
    else:        
        stagAx.plot(xSeries, yStaggered , label=file1)
    
    #stagAx.plot(xSeries, yStaggered , label=file1)
    
    #creates minor axis ticks
    stagAx.xaxis.set_minor_locator(stag_minorLocator)

    
    plt.legend(prop={'size':7})#, loc=3)
    plt.title(group)     
    
    
                                     
                                                            
                                                                                                    
def createAllPlots(path, group):   
    '''creates all the plots for the files in a particular director
    title is the title at the top of the plots'''
    global yLow, yHigh, graphs
    
    #path = str(raw_input('enter directory containing files to be plotted: '))
    
    #closes all the 14 open plots         
    for i in range(numberofPlotsDrawnLastTime):
        plt.close()
        
    #changes to whichever directory containing the data files
    os.chdir(path)
    
   
   #initializes all the figure and ax objects:
   
       #          1                2             3             4
    graphs = ['regular gr', 'staggered gr', 'regular sq', 'staggered sq']
    figures, axes = {}, {} #hash maps
    
    #initializes the fig and ax objects for all the graph types listed above
    for i in graphs:
       figures[i] = plt.figure(i, figsize=(10.0,16.0), dpi=100)
       axes[i] =  figures[i].add_subplot(111)
       setAxesTitles_Ax(axes[i], i[-2:])
       yHigh[i] = 2.0
       yLow[i] = -2.0
          
    #sets counter variable to increase the plot's height by stagger each time
    staggerCount = 0
    
    grpLabel = {}
    grpfoundDiff = {}
    grpfoundX = {}
    grpfoundY = {}

    
    #creates text file to write all the maxima to later
    new_file = str(group) + 'maxima.txt'
    f_new = open(new_file, 'w')
    
    lastFile = False
    currentFileCount = 0
    #print 'os.listdir(os.getcwd()):', os.listdir(os.getcwd())
    os.chdir(path)
    totalFileCnt = 0
    for f in os.listdir(os.getcwd()):
        if f.endswith('smt.gr'):
            totalFileCnt += 1
    #totalFileCnt = len(os.listdir(os.getcwd()))
    print ( 'totalFileCnt' , totalFileCnt) 
    #iterate through all files in directory
    for current_file in os.listdir(os.getcwd()):
        #print current_file
         
        #only looks at gr files
        if current_file.endswith("smt.gr"): 
            
            currentFileCount += 1  
            #lastFile = True
            if currentFileCount == totalFileCnt:
                lastFile = True  
            print ('Current File Count, Current File', currentFileCount, current_file)    
            
            plotData(path, 'gr', current_file, axes['regular gr'], axes['staggered gr'], group, staggerCount, f_new, grpfoundDiff, grpfoundX, grpfoundY,lastFile)
                     
            staggerCount += stagger
            
            print '\n\n\n------------------\n\n\n'
            
            continue
            
            
        #only looks at sq files
        elif current_file.endswith("smt.sq"): 
            plotData(path, 'sq', current_file, axes['regular sq'], axes['staggered sq'], group, staggerCount, f_new, grpfoundDiff, grpfoundX, grpfoundY,lastFile )
            staggerCount += staggersq
                        
            '\n\n\n------------------\n\n\n'
            
            continue
            
        else:
            continue
            
    
    #saves the files        
    #for i in ['gr', 'sq']:
#        savePlots('regular ' + i, group, path)       
#        savePlots('staggered ' + i, group, path) 
    
    #shows the matplotlib plots
    plt.show()
    
    f_new.close()




def plotAllDirs():
    '''in one function call, it can create and save plots for all 
    the directories'''
    
    #iterate through all the directories
    dirs = []
    for i in range(16, 12, -1):
        dirs.append('G' + str(i))
    dirs.append('G10')
    for i in range(7, 0, -1):
        dirs.append('G' + str(i))
    
    #print dirs
    
    for i in dirs:
    #    print ('C:\Maya\LLP\\' + i, i)   
        createAllPlots('C:\Maya\LLP\\' + i, i)


def savePlots1(fig, plotName, group, path):
    '''saves the plots as different file types for later use'''
   
    os.chdir(path)
    plt.figure(plotName)
    plt.savefig(plotName + group + '.png', format='png')
    plt.savefig(plotName+ group +'.ps', format='ps')
    plt.savefig(plotName+group +'.svg', format='svg')
    plt.savefig(plotName+group +'.pdf', format='pdf')
    plt.savefig(plotName+group +'.eps', format='eps')
    
    os.chdir('C:\Maya\LLP\png2')
    plt.savefig(plotName + ' ' + group +'.png', format='png')


# 12 directories
# G1, G2, G3, G4, G5, G6, G7, G10, G13, G14, G15, G16



group = 'G1'



#createAllPlots('C:\Maya\LLP\G1', 'G1')
createAllPlots('C:\Maya\LLP\\' + group, group)
#========================================================================
#------------------------------------------------------------------------

for i in range(30):
    plt.close()
#group = 'G1'
yBorderTop, yBorderBot = .5, .25
sqBorder = .25
stagCt = 0
stagCtsq = 0
stagGr = 6.0
stagSq = 1.5
fontlegend = 22
fontaxis = 30
fontlabel = 22
fonttitle = 40
fonttick = 18

yMaxA = {}
yMinA = {}

logYMinMax = False  # log finding Y Max and Y min values


for graphName in dtp:
   
    if graphName == 'Regular gr':   # for this graphName 
       logYMinMax = True # show detailed log finding Y Max and Y min value
    else:
       logYMinMax =  False
# uncomment the next three lines to show only selected graphs
#    if graphName == 'Regular gr':   # for this graphName 
#    else:
#       continue   # skip other graphs
       
    plotType = graphName[-2:]
    
    
    fig = plt.figure(graphName+'new', figsize=(20.0,16.0), dpi=100)
    
#    if plotType == 'gr':
#        stagCt += stagger
#    else:
#        stagCt += staggersq


    yMaxA[graphName] =  0
    yMinA[graphName] =  0
    
#    if graphName == 'H2SO4 gr':
#        verbose = True
      
    for f in dtp[graphName]:
#           print f
#           print yt[f][200:210]    
        xs[f] = np.array(xs[f])
        ys[f] = np.array(ys[f])    
        if 'Staggered' in graphName:
            plt.plot(xs[f], ys[f]+stagCt, label=f.replace(' smt',''))
            # np.nanmin -- find the min of numpy array - ingnore NaN (Not a Numbers)
            yMinA[graphName] = min(np.nanmin(ys[f])+stagCt, yMinA.get(graphName))
            yMaxA[graphName] = max(np.nanmax(ys[f])+stagCt, yMaxA.get(graphName))
            if logYMinMax:
                print 'Staggered:  max ( max(yseriesFull) + stagCt, currentMaxValue) '  \
                  , graphName, f, yMaxA[graphName], '= max(', np.nanmax(ys[f]), '+', stagCt, yMaxA.get(graphName), ')'           
                print 'Staggered:  min ( min(yseriesFull) + stagCt, currentMinValue) '  \
                  , graphName, f, yMinA[graphName], '= min(', np.nanmin(ys[f]), '+',  stagCt, yMinA.get(graphName), ')'
                
            if plotType == 'gr':
                stagCt += stagGr
            else: 
                stagCt += stagSq
        
        else:      
            plt.plot(xt[f], yt[f], label=f.replace(' smt',''))
            yMinA[graphName] = min(np.nanmin(yt[f]), yMinA.get(graphName))
            yMaxA[graphName] = max(np.nanmax(yt[f]), yMaxA.get(graphName))
            if logYMinMax:
                print 'Regular:  max ( max(yseriesTrim), currentMaxValue) '  \
                  , graphName, f, yMaxA[graphName], '= max(', np.nanmax(ys[f]), yMaxA.get(graphName), ')'           
                print 'Regular:  min ( min(yseriesTrim), currentMinValue) '  \
                  , graphName, f, yMinA[graphName], '= min(', np.nanmin(ys[f]),  yMinA.get(graphName), ')'
           
   
    print ("graphName, yMinA, yMaxA:", graphName, yMinA.get(graphName), yMaxA.get(graphName))
    
    plt.legend(prop={'size':fontlegend})
   
    setAxesTitles_plt(plotType, fontsize=fontaxis)
    plt.title(group +' '+ mixes[group[1:]] + ' ' + graphName, fontsize = fonttitle)
    
    
    
    #plots ref data, since ref data was not included in dtp
    if ref:    
        for f in ref:
            if plotType in f:            
                if 'air' in f: #plots ref air
                    
                    if 'Staggered' in graphName: #increments stagCt 
                        xs[f] = np.array(xs[f])
                        ys[f] = np.array(ys[f])                        
                        plt.plot(xs[f], ys[f]+stagCt, 'k-.', label=f.replace(' smt',''))
                        yMinA[graphName] = min(np.nanmin(ys[f])+stagCt, yMinA.get(graphName))
                        yMaxA[graphName] = max(np.nanmax(ys[f])+stagCt, yMaxA.get(graphName))
                        if logYMinMax:
                            print 'Staggered:  max ( max(yseriesFull) + stagCt, currentMaxValue) '  \
                                , graphName, f, yMaxA[graphName], '= max(', np.nanmax(ys[f]), '+', stagCt, yMaxA.get(graphName), ')'           
                            print 'Staggered:  min ( min(yseriesFull) + stagCt, currentMinValue) '  \
                                , graphName, f, yMinA[graphName], '= min(', np.nanmin(ys[f]), '+',  stagCt, yMinA.get(graphName), ')'
                        if plotType == 'gr':
                            stagCt += stagGr
                        else: 
                            stagCt += stagSq
                    else: #plots ref air on regular graphs
                        plt.plot(xt[f], yt[f], 'k-.', label=f.replace(' smt',''))
                        yMinA[graphName] = min(np.nanmin(yt[f]), yMinA.get(graphName))
                        yMaxA[graphName] = max(np.nanmax(yt[f]), yMaxA.get(graphName))
                        if logYMinMax:
                            print 'Regular:  max ( max(yseriesTrim), currentMaxValue) '  \
                              , graphName, f, yMaxA[graphName], '= max(', np.nanmax(yt[f]), yMaxA.get(graphName), ')'           
                            print 'Regular:  min ( min(yseriesTrim), currentMinValue) '  \
                              , graphName, f, yMinA[graphName], '= min(', np.nanmin(yt[f]),  yMinA.get(graphName), ')'
                                   
                
                else: #plots ref pure water
                                       
                    if 'Staggered' in graphName: #increments stagCt 
                        xs[f] = np.array(xs[f])
                        ys[f] = np.array(ys[f]) 
                        plt.plot(xs[f], ys[f]+stagCt, 'k:', label=f.replace(' smt',''))
                        yMinA[graphName] = min(np.nanmin(ys[f])+stagCt, yMinA.get(graphName))
                        yMaxA[graphName] = max(np.nanmax(ys[f])+stagCt, yMaxA.get(graphName))
                        if logYMinMax:
                            print 'Staggered:  max ( max(yseriesFull) + stagCt, currentMaxValue) '  \
                                , graphName, f, yMaxA[graphName], '= max(', np.nanmax(ys[f]), '+', stagCt, yMaxA.get(graphName), ')'           
                            print 'Staggered:  min ( min(yseriesFull) + stagCt, currentMinValue) '  \
                                , graphName, f, yMinA[graphName], '= min(', np.nanmin(ys[f]), '+',  stagCt, yMinA.get(graphName), ')'
                        if plotType == 'gr':
                            stagCt += stagGr
                        else: 
                            stagCt += stagSq                            
                    else: #plots ref pure water on regular graphs
                        plt.plot(xt[f], yt[f], 'k:', label=f.replace(' smt',''))
                        yMinA[graphName] = min(np.nanmin(yt[f]), yMinA.get(graphName))
                        yMaxA[graphName] = max(np.nanmax(yt[f]), yMaxA.get(graphName))
                        if logYMinMax:
                            print 'Regular:  max ( max(yseriesTrim), currentMaxValue) '  \
                              , graphName, f, yMaxA[graphName], '= max(', np.nanmax(yt[f]), yMaxA.get(graphName), ')'           
                            print 'Regular:  min ( min(yseriesTrim), currentMinValue) '  \
                              , graphName, f, yMinA[graphName], '= min(', np.nanmin(yt[f]),  yMinA.get(graphName), ')'
  

    
    print ("After including Reference Plots: graphName, yMinA, yMaxA:", graphName, yMinA.get(graphName), yMaxA.get(graphName))
  # need to account for different border sizes for .gr and .sq type plots -- see below
  #   print ("After including Borders: graphName, yMinA, yMaxA:", graphName, yMinA.get(graphName), yMaxA.get(graphName))
   
    if plotType=='gr':
        
        
        
        if 'Stagger' not in graphName:
            
            labelMaxima(gl, graphName+'new', fontsize=fontlabel)
            
            #not used -- replaced with yMinA and yMaxA           
#            gr_bot = []
#            for f in yBotTrim.iterkeys():
#                if f.endswith(plotType):
#                    gr_bot.append(yBotTrim[f])
#            gr_min = min(gr_bot)
#            
#            gr_top = []
#            for f in yTopTrim.iterkeys():
#                if f.endswith(plotType):
#                    gr_top.append(yTopTrim[f])
#            
#            gr_max = max(gr_top)
#            #not used -- replaced with yMinA and yMaxA
            #print 'grmin, max', gr_min, gr_max
        
           # plt.ylim(gr_min - yBorderBot, gr_max + yBorderTop)
        
            plt.ylim(yMinA[graphName] - yBorderBot, yMaxA[graphName] + yBorderTop)

        else: #cuts off x for staggered graphs
            plt.xlim(0.0, 40.0)
            
            plt.ylim(yMinA[graphName] - yBorderBot, yMaxA[graphName] + yBorderTop)
        
        plt.legend(prop={'size':fontlegend})
   
   #plt.ylim(min(yBotTrim.values())-yBorderBot, max(yTopTrim.values())+yBorderTop)
    else: #if plotType is sq
        if 'Stagger' not in graphName:
#            sq_bot = []
#            for f in yBotTrim.iterkeys():
#                if f.endswith(plotType):
#                    sq_bot.append(yBotTrim[f])
#            sq_min = min(sq_bot)
#            
#            sq_top = []
#            for f in yTopTrim.iterkeys():
#                if f.endswith(plotType):
#                    sq_top.append(yTopTrim[f])
#            
#            sq_max = max(sq_top)
#            
#            print 'sqmin, max', sq_min, sq_max
        
            #plt.ylim(sq_min - sqBorder, sq_max + sqBorder)
            plt.ylim(yMinA[graphName] - sqBorder, yMaxA[graphName] + sqBorder)   
        
        plt.legend(prop={'size':fontlegend}, loc=4)

    atp[graphName] = fig.add_subplot(111)
    atp[graphName].xaxis.set_minor_locator(reg_minorLocator)
    atp[graphName].tick_params(axis='both', which='major', labelsize=fonttick)
    #fig.savefig(graphName+'.png')    
    #savePlots(fig, graphName, group, 'C:\Maya\LLP\G1')
 #   plt.savefig(graphName+'.eps', format='eps')
 #  p = "C:\Maya\LLP\Figs\G1\\"   
    p = 'C:\Maya\LLP\Figs\\' + group + "\\"
 #   os.chdir(path)
    pathFile =  p +  graphName
    # set up a figure object before showing or drawing
        # so that later on you can save it -- in a loop show sets up a new figure
    fig1 = plt.gcf()  # Importan: set up a figure object before showing or drawing
    plt.show()
    plt.draw()
    fig1.savefig(pathFile+'.png', format='png')   
    fig1.savefig(pathFile+'.eps', format='eps')  

 #   plt.savefig(pathFile, format='png')
    
#==============================================================================
#     plt.figure(graphName)
#     plt.savefig(pathFile, format='png')
#     plt.show()
#     
#==============================================================================
#==============================================================================
#     fig1 = plt.gcf()
#     plt.show()
#     plt.draw()
#     fig1.savefig('tessstttyyy.png', dpi=100)
#==============================================================================

lcx = {}
lcy = {}


#import csv


os.chdir('C:\Maya\LLP\G2aptgraphs')

import pandas
import xlsxwriter

#gCompounds = []
#gCompoundsXY = []
i = 0

df_all = {}
curr_l = []
#if i <1:

#p = "C:\Maya\LLP\Figs\G1\\excel\\"
p = "C:\Maya\LLP\Figs\\" + group + "\\excel\\"
 #   os.chdir(path)
 
for graphName in dtp:
    #i+=1
    if 'sq' not in graphName:
        gCompounds = []
        gCompoundsXY = []  


        # Firs pass, determine the DataFrame, set up DataFrame 
        # Calculate dimensions, rows header, column headers

        #numFiles=len(dtp[graphName]) 
        #graphName = "Na2SO4 gr" # for testing, otherewise comment   
        allfiles = []
        print ("allfiles at start up: ", allfiles)
        allfiles =  copy.deepcopy(dtp[graphName])
        print ("allfiles after copying from dtp: ",  allfiles)
        allfiles.extend(ref)
        print ("allfiles after extend : ",  allfiles)
        print ("allfiles", allfiles)
        numFiles=len(dtp[graphName])
        
        #for f in dtp[graphName]:
        for f in allfiles:  
            print 'graphName, file, numfiles: ', graphName, f, numFiles
            curr_l = []
            curr_l=copy.deepcopy(l.get(f))
            #curr_l.append(l[f])
            
            #print curr_l
            compound_values = {}  
            if curr_l:            
                compound_values = {v2.replace('\n',''): v1 for v1, v2 in curr_l.iteritems()}
 
        for c in compound_values.iterkeys():        
            if c not in gCompounds:            
                gCompounds.append(c)
            
        # gCompoundsXY = []   # used when testing, comment out when in main loop
        for c in compound_values.iterkeys():  
            cX = c + " x"
            cY = c + " y"
            if cX not in gCompounds:            
                gCompoundsXY.append(cX)
            if cY not in gCompounds: 
                gCompoundsXY.append(cY)
            
#    for f1 in dtp[graphName].iterkeys():        
#        if f1 not in gCompounds:            
#            gFiles.append(f1)
    
        nCompounds = len(gCompounds)
        index = []
        #graphName = 'Na2SO4 gr'
    #df=pandas.DataFrame(index=['A','B','C'], columns=['x','y'])
#  set up Dataframee
        df1=pandas.DataFrame(index=dtp[graphName], columns=gCompounds)
        df2=pandas.DataFrame(index=dtp[graphName], columns=gCompoundsXY)
        verbose = False
    #   seconcd pass  
        print ("Starting secnond pass, setting up cell values")
        #for f in dtp[graphName]:
        for f in allfiles:
            if verbose: 
                print("graphName: ", graphName, "file: ", f)
            curr_l = []
            curr_l=copy.deepcopy(l.get(f))
            #curr_l.append(l[f])
            if verbose:
                print ("Current Peak Values", curr_l)
                print ("Switching Key - Value Pairs around")
            compound_values = {}  
            print ('curr_l before key-value switch :', curr_l)
            if curr_l:
                compound_values = {v2.replace('\n',''): v1 for v1, v2 in curr_l.iteritems()}
            if verbose:
                print ("Compound Values", compound_values )
            for xyc in compound_values.iteritems():
                if verbose:
                   print ('f, xyc: ', f, xyc)
                compoundval, xyval = xyc 
                x, y  = xyval
                xyvalStr = str(x) + " " + str(y)
                if verbose:
                    print ("Compound, (X,Y)", compoundval, xyvalStr) 
                df1.set_value(f, compoundval, xyvalStr)   
     
                compoundNameX = compoundval + " x"
                compoundNameY = compoundval + " y"
                df2.set_value(f, compoundNameX, x)  
                df2.set_value(f, compoundNameY, y)
                 
            #print ("df1","\n", df1)
            #print ("df2","\n", df2)
      # there will be only one Excel per graphName (that is not .sq)      
      # Create a Pandas Excel writer using XlsxWriter as the engine.
      #  import xlsxwriter
        pathFile =  p +  graphName  
        #  pathFile =  p + group + " " + graphName  
        
        writer = pandas.ExcelWriter(pathFile+".xlsx",engine='xlsxwriter')
        df1.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        writer.close()
        # write df1 with separate columns for x and y -- merge header by hand
        writer = pandas.ExcelWriter(pathFile+" xy " + ".xlsx",engine='xlsxwriter')
        df2.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        writer.close()
        
#==============================================================================
# len(dtp['Regular gr'])
#             
#     
#     ofile  = open(graphName+'.csv', "wb")
#     writer = csv.writer(ofile, delimiter='	')#, quotechar='"', quoting=csv.QUOTE_ALL)
#     
#     col_headers = []
#     for label in gl.iteritems():
#         col_headers.append(label)        
#     writer.writerow(label)
#     
#     ofile.close()
#==============================================================================

    
    
       
       
       
       
#       print min(yBotTrim.values())
#       print max(yTopTrim.values())
   
#==============================================================================
#==============================================================================
 # #==============================================================================
 # # for graphName in dtp:
 # #     #plt.figure(graphName+'new')
 # #     for f in dtp[graphName]:
 # #         print f
 # #         print yt[f][200:210]        
 # #         #plt.plot(xt[f], yt[f])#, label=f)
 # #==============================================================================
#==============================================================================
#==============================================================================
        
#==============================================================================
# #    for i in aptgraphs:
# #        plt.figure(i)
# #        
# #        if 'Ref' in file1:
# #            if 'air' in file1:
# #                plt.plot(xSeriesTrim, ySeriesTrim, 'k-.', label=file1)
# #            else:
# #                plt.plot(xSeriesTrim, ySeriesTrim, 'k:', label=file1)
# #        else:        
# #            plt.plot(xSeriesTrim, ySeriesTrim , label=file1)
# #        
# #        plt.legend(prop={'size':7})
# #        setAxesTitles_plt(i[-2:])
#==============================================================================
