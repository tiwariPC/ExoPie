import os 
import sys 
#import optparse
import argparse
import numpy
import pandas
from root_pandas import read_root
from pandas import  DataFrame, concat
from pandas import Series 
from ROOT import TLorentzVector, TH1F
import time

## snippet from Deepak
from multiprocessing import Process
import multiprocessing as mp


sys.path.append('utils')
sys.path.append('../utils')
from MathUtils import * 


sys.path.append('../analysistools/xsectools/')
sys.path.append('analysistools/xsectools/')
from PlotParameters import xs, xsweight
#getPt, getEta, getPhi, logical_AND, logical_OR, getMT, Delta_R, logical_AND_List*
#from pillar import *


print "starting clock"
start = time.clock()


debug_=False

usage = "analyzer for t+DM (debugging) "
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-i", "--inputfile",  dest="inputfile")
parser.add_argument("-o", "--outputfile", dest="outputfile",default="out.root")
parser.add_argument("-D", "--outputdir", dest="outputdir")
parser.add_argument("-F", "--farmout", action="store_true",  dest="farmout")
args = parser.parse_args()

infile = args.inputfile
print 'outfile= ', args.outputfile

'''
## https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6
## https://github.com/deepakcern/MonoH/blob/monoH_boosted/bbDM/bbDM/bbMET/SkimTree.py

Tutorial links 
pandas full official tutorial: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
https://lhcb.github.io/analysis-essentials/python/first_histogram.html
https://stackoverflow.com/questions/43772362/how-to-print-a-specific-row-of-a-pandas-dataframe
https://github.com/hungapl/python/blob/master/performance/iterate-performance-comparison.py
https://towardsdatascience.com/different-ways-to-iterate-over-rows-in-a-pandas-dataframe-performance-comparison-dc0d5dcef8fe
Raman reported issue: https://github.com/scikit-hep/root_numpy/issues/390 
can be further fasten by using multiprocessing: tutorial at: http://gouthamanbalaraman.com/blog/distributed-processing-pandas.html

This tutorial help is makeing dataframe with only those variables for which we need to make the histograms. This dataframe can be tried to save into .rootfile as this will have very small number of events. And then later on analyze using matplotlib in future. For now only histograms of this dataframe will be saved. 
https://thispointer.com/pandas-how-to-create-an-empty-dataframe-and-append-rows-columns-to-it-in-python/

In order to make stack plots save the event weights properly and then following tutorial can be used. 

Stack plot tutorial in matplotlibn: https://www.youtube.com/watch?v=Z81JW1NTsO8

For ratio plot and two pads 

style, ststs and systematics still missing 
'''




def runtdm(infile_):
    print "running the code for ",infile_
    tag_str = infile_.split("_13TeV")[0].split("Merged_")[1]
    
    cross_section_weight = xsweight(tag_str)
    
    print infile_, tag_str, cross_section_weight
    
    ### dataframe for output 
    df_out = DataFrame(columns=['run', 'lumi', 'event', 'MET', 'MT', 'Njets_PassID', 'Nbjets_PassID','ElePt', 'EleEta', 'ElePhi', 'Jet1Pt', 'Jet1Eta', 'Jet1Phi', 
                                'Jet2Pt','Jet2Eta', 'Jet2Phi', 'Jet3Pt','Jet3Eta','Jet3Phi','Jet1Idx','Jet2Idx','Jet3Idx', 'weight'])
    df_out_wmunu_cr = DataFrame(columns=['run', 'lumi', 'event', 'MET', 'MT', 'Njets_PassID', 'Nbjets_PassID', 'Jet1Pt', 'Jet1Eta', 'Jet1Phi',
                                         'Jet2Pt','Jet2Eta', 'Jet2Phi', 'Jet3Pt','Jet3Eta','Jet3Phi', 'Jet1Idx', 'Jet2Idx', 'Jet3Idx', 
                                         'MuPt', 'MuEta', 'MuPhi','weight'])
    
    
    recoil_den  = TH1F("recoil_den","recoil_den", 100, 0.0,1000.)
    recoil_num  = TH1F("recoil_num","recoil_num", 100, 0.0,1000.)
    
    

    jetvariables = ['st_THINnJet','st_THINjetPx','st_THINjetPy','st_THINjetPz','st_THINjetEnergy', 'st_THINjetCISVV2','st_THINjetHadronFlavor','st_THINjetNHadEF','st_THINjetCHadEF','st_THINjetCEmEF','st_THINjetPhoEF','st_THINjetEleEF','st_THINjetMuoEF','st_THINjetCorrUnc','st_runId','st_lumiSection','st_eventId','st_pfMetCorrPt','st_pfMetCorrPhi','st_pfMetUncJetResUp','st_pfMetUncJetResDown','st_pfMetUncJetEnUp','st_pfMetUncJetEnDown','st_isData','st_HLT_IsoMu24_v','st_HLT_IsoTkMu24_v','st_HLT_IsoMu27_v','st_HLT_IsoTkMu27_v','st_HLT_Ele27_WPTight_Gsf_v','st_HLT_Ele27_WPLoose_Gsf_v','st_HLT_Ele105_CaloIdVT_GsfTrkIdT_v','st_HLT_Ele115_CaloIdVT_GsfTrkIdT_v','st_HLT_Ele32_WPTight_Gsf_v','st_HLT_Ele32_eta2p1_WPTight_Gsf_v','st_HLT_Ele27_eta2p1_WPTight_Gsf_v','st_THINnJet','st_THINjetPx','st_THINjetPy','st_THINjetPz','st_THINjetEnergy','st_THINjetCISVV2','st_THINjetHadronFlavor','st_THINjetNHadEF','st_THINjetCHadEF','st_THINjetCEmEF','st_THINjetPhoEF','st_THINjetEleEF','st_THINjetMuoEF','st_THINjetCorrUnc','st_nEle','st_elePx','st_elePy','st_elePz','st_eleEnergy','st_eleIsPassTight','st_nMu','st_muPx','st_muPy','st_muPz','st_muEnergy','st_isTightMuon','st_muIso']

    filename = infile_


    ''' global variables, mainly to be stored in the new root tree for quick analysis or histo saving '''
    
    icount=0
    
    df_new  = DataFrame()
    
    df_all  = DataFrame()
    
    ieve=0
    
    jetptseries=[]
    jetetaseries=[]
    jetphiseries=[]
    jet_pt30=[]
    jet_pt50=[]
    jet_eta4p5=[]
    jet_IDtightVeto=[]
    jet_eta2p4=[]
    jet_NJpt30=[]
    jet_NJpt30_Eta4p5=[]
    jet_csvmedium=[]
    jet_N_bmedium_eta2p4_pt30=[]
    
    hlt_ele=[]
    elept_=[]
    eleeta_=[]
    elephi_=[]
    ele_pt30_=[]
    ele_IDTight_=[]
    ele_eta2p1_=[]
    mt_ele_=[]
    mt_mu_=[]
    
    Nmu_=[]
    mupt_=[]
    mueta_=[]
    muphi_=[]
    mu_IDTight_=[]
    mu_IsoTight_=[]
    mu_pt30_=[]
    
    met_250_=[]
    for df in read_root(filename,columns=jetvariables, chunksize=125000):
        icount=icount+1
        ''' all the operations which should be applied to each event must be done under this loop, 
        otherwise effect will be reflected on the last chunk only. Each chunck can be considered 
        as a small rootfile. An example of how to add new variable and copy the new dataframe into 
        a bigger dataframe is shown below. This is the by far fastest method I manage to find, uproot
        awkward arrays are even faster but difficult to use on lxplus. and may be on condor.  ''' 
        for nak4jet_, ak4px_, ak4py_, ak4pz_, ak4e_, ak4csv, ak4flavor, ak4NHEF, ak4CHEF, ak4CEmEF, ak4PhEF, ek4EleEF, ak4MuEF, ak4JEC, hlt_ele27, hlt_ele105, hlt_ele115, hlt_ele32, hlt_ele32_eta2p1, hlt_ele27_eta2p1, nele_, elepx_, elepy_, elepz_, elee_, eletightid_, nmu_, mupx_, mupy_, mupz_, mue_, mutightid_, muIso_, met_, metphi_, run, lumi, event in zip(df.st_THINnJet, df.st_THINjetPx, df.st_THINjetPy, df.st_THINjetPz, df.st_THINjetEnergy, df.st_THINjetCISVV2, df.st_THINjetHadronFlavor, df.st_THINjetNHadEF, df.st_THINjetCHadEF, df.st_THINjetCEmEF, df.st_THINjetPhoEF, df.st_THINjetEleEF, df.st_THINjetMuoEF, df.st_THINjetCorrUnc, df.st_HLT_Ele27_WPLoose_Gsf_v, df.st_HLT_Ele105_CaloIdVT_GsfTrkIdT_v, df.st_HLT_Ele115_CaloIdVT_GsfTrkIdT_v, df.st_HLT_Ele32_WPTight_Gsf_v, df.st_HLT_Ele32_eta2p1_WPTight_Gsf_v, df.st_HLT_Ele27_eta2p1_WPTight_Gsf_v, df.st_nEle, df.st_elePx, df.st_elePy, df.st_elePz, df.st_eleEnergy, df.st_eleIsPassTight, df.st_nMu, df.st_muPx, df.st_muPy, df.st_muPz, df.st_muEnergy, df.st_isTightMuon, df.st_muIso, df.st_pfMetCorrPt, df.st_pfMetCorrPhi, df.st_runId, df.st_lumiSection, df.st_eventId):
            
            if debug_: print "ievent = ",ieve

            ieve=ieve+1
            if debug_: print nak4jet_, ak4px_, ak4py_, ak4pz_, ak4e_
            
            '''
            *******   *****   *******
            *      *          *
            *      ****       *
            *      *          *
            ***        *****      *
            '''
            
            ''' This small block compute the pt of the jet and add them back 
            into the original dataframe as a next step for further usage. '''
            ak4pt = [getPt(ak4px_[ij], ak4py_[ij]) for ij in range(nak4jet_)]
            #print ak4pt
            jetptseries.append(ak4pt)
            
            
            ''' Jet Loose ID is already applied in the preselection  ''' 
            
            ''' eta and phi of the ak4 jets '''
            ak4eta = [getEta(ak4px_[ij], ak4py_[ij], ak4pz_[ij]) for ij in range(nak4jet_)]
            ak4phi = [getPhi(ak4px_[ij], ak4py_[ij] ) for ij in range(nak4jet_)]
            
            jetetaseries.append(ak4eta)
            jetphiseries.append(ak4phi)
            
            ''' pT>30 GeV, |eta|<4.5 is already applied in the tuples '''
            
            ''' jets with pt > 30 GeV '''
            ak4_pt30 = [(ak4pt[ij] > 30.)  for ij in range(nak4jet_)]
            jet_pt30.append(ak4_pt30)
            
            ''' jets with pt > 50 GeV '''
            ak4_pt50 = [(ak4pt[ij] > 50.)  for ij in range(nak4jet_)]
            jet_pt50.append(ak4_pt50)
            
            ''' jet |eta| < 4.5  '''
            ak4_eta4p5 = [(abs(ak4eta[ij]) < 4.5)  for ij in range(nak4jet_)]
            jet_eta4p5.append(ak4_eta4p5)
            
            ''' jet |eta| < 2.4 '''
            ak4_eta2p4 = [(abs(ak4eta[ij]) < 2.4)  for ij in range(nak4jet_)]
            jet_eta2p4.append(ak4_eta2p4)
            
            ''' jet tightLeptonVeto ID to reject fake jets coming from the leptons, Veto ID should be applied only for jets within the detector abs(eta) < 2.4
            
            Following the syntax of if else in list comprehension 
            [f(x) if condition else g(x) for x in sequence]
            '''
            
            ak4_IDtightVeto = [ ( (ak4NHEF[ij]<0.90) and (ak4PhEF[ij]<0.90)  and (ak4MuEF[ij]<0.8)  and (ak4CEmEF[ij]<0.90) and abs(ak4eta[ij]) < 2.4 ) 
                                or ( (ak4NHEF[ij]<0.90) and (ak4PhEF[ij]<0.90) and (ak4MuEF[ij]<0.8)  and abs(ak4eta[ij]) < 2.7  and abs(ak4eta[ij]) > 2.4 ) if (abs(ak4eta[ij]) < 2.7)
                                else True for ij in range(nak4jet_) ]
            jet_IDtightVeto.append(ak4_IDtightVeto)
            
            if debug_:  print "ak4_IDtightVeto", ak4_IDtightVeto
            
            ''' njets passing jet pt > 30 and eta < 4.5 and Loose Jet ID '''
            jet_NJpt30.append(ak4_pt30.count(True))
            
            jet_NJpt30_Eta4p5.append(len([ij  for ij in range(nak4jet_) if (ak4_eta4p5[ij] and ak4_pt50[ij]) ]) )
            
            ak4_csvmedium = [(ak4csv[ij] > 0.8484) for ij in range(nak4jet_)]
            jet_csvmedium.append(ak4_csvmedium)
            
            jet_N_bmedium_eta2p4_pt30.append(len([ ij for  ij in range(nak4jet_) if ( (ak4_eta2p4[ij]) and (ak4_pt30[ij]) and (ak4_csvmedium[ij])   )  ] ) )
            
            
            
            '''
            
            ****   *      ****
            *      *      *
            ***    *      ***
            *      *      *
            ****   ****   ****
            
            the selection for the electron is done here, later the new branches are added to the dataframe. 
            
            '''
            
            
            ''' electron triggers ''' 
            hlt_ele.append( logical_OR([hlt_ele27, hlt_ele105, hlt_ele115, hlt_ele32, hlt_ele32_eta2p1, hlt_ele27_eta2p1]) )
            
            if debug_:  print "event ------", event
            ''' get pt, eta, phi of electrons '''
            elept  = [getPt(elepx_[ie], elepy_[ie]) for ie in range(nele_)]
            eleeta = [getEta(elepx_[ie], elepy_[ie], elepz_[ie]) for ie in range(nele_)]
            elephi = [getPhi(elepx_[ie], elepy_[ie]) for ie in range(nele_)]
            ''' electron pt and eta cut, tuples already have electron pT > 10 GeV and |eta|<2.5
            Loose electron ID is also applied on the electron at preselection level '''
            ele_pt30  =[(elept[ie] > 30)  for ie in range(nele_)]
            ele_IDTight = [(eletightid_[ie])  for ie in range(nele_)]
            ele_eta2p1 = [(abs(eleeta[ie]) < 2.1)  for ie in range(nele_)]
            
            if debug_:  print "ele info"
            if debug_:  print "pt, id eta =", ele_pt30, ele_IDTight, ele_eta2p1
            if debug_:  
                for ie in range(nele_): print elept[ie], eleeta[ie], eletightid_[ie], elepx_[ie], elepy_[ie], elepz_[ie]
            #print "debuging", ele_eta2p1, ele_IDTight, ele_pt30 
            #print ele_pt30, ele_IDTight, ele_eta2p1
            elept_.append(elept)
            eleeta_.append(eleeta)
            elephi_.append(elephi)
            ele_pt30_.append(ele_pt30)
            ele_IDTight_.append(ele_IDTight)
            ele_eta2p1_.append(ele_eta2p1)
            
            
            ''' electron ID and Isolation, '''
            
            '''
            
            **     *  *     *
            * *  * *  *     *
            *  *   *  *     *
            *      *  *     *
            *      *   ***** 
            
            the selection for the muon is done here, later the new columns are added to the dataframe for each of them. 
            
            '''
            
            ''' muon triggers '''
            
            ''' muon pt threshold and eta threshold, tuples already have muon pt > 10 and |eta| < 2.4 '''
            mupt  = [getPt(mupx_[imu], mupy_[imu]) for imu in range(nmu_)]
            mueta = [getEta(mupx_[imu], mupy_[imu], mupz_[imu]) for imu in range(nmu_)]
            muphi = [getPhi(mupx_[imu], mupy_[imu]) for imu in range(nmu_)]
            
            ''' For vetoing in the electron region only Looose Mu ID and ISo with pt > 10 GeV is needed and is already applied in the skimmer '''
            mu_pt30 = [ (mupt[imu] > 30.0) for imu in range(nmu_)]
            mu_IDTight  = [mutightid_[imu] for imu in range(nmu_)]
            mu_IsoTight = [(muIso_[imu]<0.15) for imu in range(nmu_)]
            
            
            Nmu_.append(nmu_)
            mupt_.append(mupt)
            mueta_.append(mueta)
            muphi_.append(muphi)
            
            mu_IDTight_.append(mu_IDTight)
            mu_IsoTight_.append(mu_IsoTight)
            mu_pt30_.append(mu_pt30)
            
            
            
            
            ''' MET SELECTION '''
            met_250_.append( met_ > 250.0 )
            
            
            ''' MT Selection ''' 
            mt_ele =  [ getMT(elept[ie], met_, elephi[ie], metphi_) for ie in range(nele_)   ]
            mt_ele_.append(mt_ele)
            
            
            ''' MT Selection ''' 
            mt_mu =  [ getMT(mupt[imu], met_, muphi[imu], metphi_) for imu in range(nmu_)   ]
            mt_mu_.append(mt_mu)
            
            ''' 
            Event selection to count the number of events. 
            
            In simple terms, index() method finds the given element in a list and returns its position.
            However, if the same element is present more than once, index() method returns its smallest/first position. 
            And this is what I generally need for this code. But this fails when there is no element or no true in the list 
            
            This is complicated in first look but more usable. And it will be faster once I know how to flatten the dataset. 
            
            first elecment of output return by where is the location whre true is present, still not known how where actually work but this is the fastest method 
            e.g. 
            #if (len(ele_eta2p1)>0): ele_passlist = numpy.where(ele_eta2p1)[0]
            #print ele_passlist
            '''
            
            ''' take AND of all the electron cuts (just take the lists) '''
            ele_eta2p1_idT_pt30=[]
            if (len(ele_eta2p1)>0):             ele_eta2p1_idT_pt30 = logical_AND_List3(ele_eta2p1, ele_IDTight, ele_pt30)
            
            ''' 
            > 0 means >= 1. The selection in the function is implemented like >= not >. Therefore pay attention when using this function. 
            The function also take care of the fact that the operation will be performed only when size of the list is >= N, where N is by default 0 and has to be provided 
            '''
            pass_ele_index = WhereIsTrue(ele_eta2p1_idT_pt30, 1) 
            
            mu_eta2p4_idT_pt30=[]
            if (len(mu_pt30)>0):              mu_eta2p4_idT_pt30 = logical_AND_List3(mu_pt30, mu_IDTight, mu_IsoTight)
            
            pass_mu_index = WhereIsTrue(mu_eta2p4_idT_pt30,1)
            
            ak4_pt30_eta4p5_IDL=[]
            if len(ak4_pt30)>0:             ak4_pt30_eta4p5_IDL = logical_AND_List3(ak4_pt30, ak4_eta4p5, ak4_IDtightVeto)
            
            ''' we need at least 3 jets passing id, so we must ensure presene of 3 jets atleast '''
            pass_jet_index= WhereIsTrue(ak4_pt30_eta4p5_IDL, 3)
            
            
            ak4_bjetM_eta2p4= []
            if len(ak4_csvmedium)>0:         ak4_bjetM_eta2p4 = logical_AND_List3 (ak4_csvmedium, ak4_eta2p4, ak4_IDtightVeto)
            
            pass_bjetM_eta2p4_index= WhereIsTrue(ak4_bjetM_eta2p4, 1)
            
            ''' 
            
            All the object selection is done before this, 
            region specific cuts are here. 
            
            '''
            
            pass_jet_cleaned_index_ = pass_jet_index
            ''' jet cleaning '''
            ## jet cleaning is switched off for now, because the TightLeptonVeto Jet ID is being used. 
            ## pass_jet_cleaned_index_ = [pass_jet_index[ij] for ij in range(len(pass_jet_index)) if Delta_R( ak4eta[ij], eleeta[eleidx], ak4phi[ij], elephi[eleidx]) > 0.4 ]
            
            ''' now apply all the selection '''
            if (len(pass_ele_index) >0) and (len(pass_jet_cleaned_index_)>=3):
                
                eleidx = pass_ele_index[0]
                j1idx  = pass_jet_cleaned_index_[0]
                j2idx  = pass_jet_cleaned_index_[1]
                j3idx  = pass_jet_cleaned_index_[2]
                
            
                wenu_cr = logical_AND( [len(pass_ele_index)==1, nmu_==0, met_>250.0, len(pass_jet_index)>=3, len(pass_bjetM_eta2p4_index)==0, mt_ele[pass_ele_index[0]]<160.]) #, Delta_R(ak4eta[j1idx], eleeta[eleidx],ak4phi[j1idx], elephi[eleidx]) > 0.4 , Delta_R(ak4eta[j2idx], eleeta[eleidx],ak4phi[j2idx], elephi[eleidx]) > 0.4 , Delta_R(ak4eta[j3idx], eleeta[eleidx],ak4phi[j3idx], elephi[eleidx]) > 0.4  ])
                
                #print "Wjet selection =", len(pass_ele_index)==1, nmu_==0, met_>250.0, len(pass_jet_index)>=3, len(pass_bjetM_eta2p4_index)==0, mt_ele[pass_ele_index[0]]<160., wenu_cr
                if debug_:  print "object info", wenu_cr,  run, lumi, event, eleidx, elept[eleidx], eleeta[eleidx], elephi[eleidx], j1idx, ak4pt[j1idx], ak4eta[j1idx], ak4phi[j1idx], j2idx, ak4pt[j2idx], ak4eta[j2idx], ak4phi[j2idx], j3idx, ak4pt[j3idx], ak4eta[j3idx], ak4phi[j3idx], met_, mt_ele[pass_ele_index[0]], [len(pass_ele_index)==1, nmu_==0, met_>250.0, len(pass_jet_index)>=3, len(pass_bjetM_eta2p4_index)==0, mt_ele[pass_ele_index[0]]<160.]
                #print wenu_cr, run, lumi, event
                
                
                ''' converti this infor a function '''
                if wenu_cr:
                    df_out = df_out.append({'run':run, 'lumi':lumi, 'event':event, 
                                            'MET': met_, 'MT': mt_ele[pass_ele_index[0]], 'Njets_PassID': len(pass_jet_cleaned_index_), 'Nbjets_PassID':len(pass_bjetM_eta2p4_index), 
                                            'ElePt':elept[eleidx], 'EleEta':eleeta[eleidx], 'ElePhi':elephi[eleidx], 'Jet1Pt':ak4pt[j1idx], 'Jet1Eta':ak4eta[j1idx], 'Jet1Phi':ak4phi[j1idx],
                                            'Jet2Pt':ak4pt[j2idx],'Jet2Eta':ak4eta[j2idx], 'Jet2Phi':ak4phi[j2idx], 'Jet3Pt':ak4pt[j3idx],'Jet3Eta':ak4eta[j3idx],'Jet3Phi':ak4phi[j3idx], 
                                            'Jet1Idx':j1idx, 'Jet2Idx':j2idx,'Jet3Idx':j3idx,'weight':cross_section_weight }, ignore_index=True)
                    
                    
                    
                    
                    

            ''' W mu nu CR ''' 
            pass_jet_cleaned_index_ = pass_jet_index
            
            if ( len(pass_mu_index) > 0) and (len(pass_jet_index)>=3):
                muidx = pass_mu_index[0]
                ''' right now we are assuming that leptonveto jet id is enough for cleaning so no more cleaning required '''
                j1idx  = pass_jet_cleaned_index_[0]
                j2idx  = pass_jet_cleaned_index_[1]
                j3idx  = pass_jet_cleaned_index_[2]
                
                wmunu_cr = logical_AND( [len(pass_mu_index)==1, nele_==0, met_>250.0, len(pass_jet_index)>=3, len(pass_bjetM_eta2p4_index)==0, mt_mu[muidx]<160.])
                
                if debug_:  print "object info mu ", wmunu_cr,  run, lumi, event, mupt[muidx], mueta[muidx], muphi[muidx], j1idx, ak4pt[j1idx], ak4eta[j1idx], ak4phi[j1idx], j2idx, ak4pt[j2idx], ak4eta[j2idx], ak4phi[j2idx], j3idx, ak4pt[j3idx], ak4eta[j3idx], ak4phi[j3idx], met_, mt_mu[pass_mu_index[0]], [len(pass_mu_index)==1, nmu_==1, met_>250.0, len(pass_jet_index)>=3, len(pass_bjetM_eta2p4_index)==0, mt_mu[pass_mu_index[0]]<160.]
                if wmunu_cr:
                    df_out_wmunu_cr = df_out_wmunu_cr.append({'run':run, 'lumi':lumi, 'event':event, 
                                                              'MET': met_, 'MT': mt_mu[muidx], 'Njets_PassID': len(pass_jet_cleaned_index_), 'Nbjets_PassID':len(pass_bjetM_eta2p4_index),
                                                              'Jet1Pt':ak4pt[j1idx], 'Jet1Eta':ak4eta[j1idx], 'Jet1Phi':ak4phi[j1idx],
                                                              'Jet2Pt':ak4pt[j2idx],'Jet2Eta':ak4eta[j2idx], 'Jet2Phi':ak4phi[j2idx], 'Jet3Pt':ak4pt[j3idx],'Jet3Eta':ak4eta[j3idx],
                                                              'Jet3Phi':ak4phi[j3idx],  'Jet1Idx':j1idx, 'Jet2Idx':j2idx,'Jet3Idx':j3idx, 'MuPt':mupt[muidx], 'MuEta':mueta[muidx], 
                                                              'MuPhi':muphi[muidx],'weight':cross_section_weight }, 
                                                             ignore_index=True)
                    
        ''' the dataframe must be merged after each chunksize 
        Following code must be in the dataframe chunksize loop, 
        otherwise it will be completely wrong. '''
        
        ''' right now we don't need all these big Series in the dataframe, we can save them in the rootfile and plot ''' 
        
        '''
        df['ak4pt_'] = Series(jetptseries)
        df['ak4eta_'] = Series(jetetaseries)
        df['ak4phi_'] = Series(jetphiseries)
        df['ak4_pt30_'] = Series(jet_pt30)
        df['ak4_pt50_'] = Series(jet_pt50)
        df['ak4_eta4p5'] = Series(jet_eta4p5)
        df['ak4_eta2p4'] = Series(jet_eta2p4)
        df['ak4_NJ_pt30'] = Series(jet_NJpt30)
        df['ak4_NJ_pt30_eta4p5'] = Series(jet_NJpt30_Eta4p5)
        
        df['hlt_ele'] = Series(hlt_ele)
        df['elept_'] = Series(elept_)
        df['eleeta_'] = Series(eleeta_)
        df['elephi_'] = Series(elephi_)
        df['ele_pt30_'] = Series(ele_pt30_)
        df['ele_IDTight_'] = Series(ele_IDTight_)
        df['ele_eta2p1_'] = Series(ele_eta2p1_)
        
        
        df['nmu_'] = Series(nmu_)
        df['mupt_'] = Series(mupt_)
        df['mueta_'] = Series(mueta_)
        df['muphi_'] = Series(muphi_)
        
        df['met_250_'] = Series(met_250_)
        
        '''
        
        #df[''] = Series()
        
        ''' 
        
        Join the new branched to alredy existing dataframe 
        
        '''
        
        #df[] = Series()
        df_all = concat([df_all,df])
    
    if debug_:     print df_out    
    
    outputfilename = args.outputfile
    df_out.to_root(outputfilename, key='t_dm_wenucr')
    
    df_out_wmunu_cr.to_root(outputfilename, key='t_dm_wmunucr',mode='a')
    

    
    end = time.clock()
    print "%.4gs" % (end-start)

files=['/tmp/khurana/ExoPieInput_tDM_06052019/Merged_WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_MC25ns_LegacyMC_20170328_0000_1.root', '/tmp/khurana/ExoPieInput_tDM_06052019/Merged_WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_MC25ns_LegacyMC_20170328_0000_2.root']
if __name__ == '__main__':
    try:
        pool = mp.Pool(4)
        pool.map(runtdm,files)
        pool.close()
    except Exception as e:
        print e
        pass
        
        






