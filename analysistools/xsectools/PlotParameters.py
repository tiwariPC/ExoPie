from pandas import  DataFrame, concat, read_csv
from os.path import isfile, join
import os
#from os import  getcwd, chdir

''' 
following code is just to get the path of the cross-sectoin table, full path is needed so that this code can be used from any path. This can be avoided in future by writing a single .txt / .json 
file to get all these kind of informations 

keepoing this only for syntax, in case needed later on 

cwdi =  os.getcwd()
print cwdi
os.chdir('../..')
cwd = os.getcwd()
print cwd
os.chdir(cwdi)

binpath=join(cwd, 'bin')

tablepath = join (binpath, 'crosssectionTable.txt')



 table path set end here
''' 

## above code doesn't work if this file is used in diff dir, therefore hardsetting the path, 
tablepath='/afs/cern.ch/work/k/khurana/Run2Legacy/CMSSW_10_3_0/src/ExoPie/bin/crosssectionTable.txt'


def xs(sample_tag):
    df = DataFrame(columns=['tag_str', 'rootfileName', 'total_events', 'xsec', 'xsec_weight', 'Legend'])
    df = read_csv(tablepath, sep = " ")
    
    ''' this convert the first column as index of the dataframe so that dataframe row can be accessed using this, as a key value '''
    df = df.set_index(["tag_str"])
    return df.loc[sample_tag,'xsec']


def xsweight(sample_tag):
    df = DataFrame(columns=['tag_str', 'rootfileName', 'total_events', 'xsec', 'xsec_weight', 'Legend'])
    df = read_csv(tablepath, sep = " ")
    
    ''' this convert the first column as index of the dataframe so that dataframe row can be accessed using this, as a key value '''
    df = df.set_index(["tag_str"])
    return df.loc[sample_tag,'xsec_weight']
    



''' uncomment following to test the code '''
#print xs("WJetsToLNu_HT-2500ToInf_TuneCUETP8M1"), xsweight("WJetsToLNu_HT-2500ToInf_TuneCUETP8M1")
