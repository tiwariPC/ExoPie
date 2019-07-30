import os
import sys
#import optparse
import argparse
import traceback
import numpy
import pandas
from root_pandas import read_root
from pandas import  DataFrame, concat
from pandas import Series
from ROOT import TLorentzVector, TH1F
import time

# snippet from Deepak
from multiprocessing import Process
import multiprocessing as mp

sys.path.append('utils')
sys.path.append('../utils')
from MathUtils import *

#sys.path.append('analysistools/xsectools/')
# getPt, getEta, getPhi, logical_AND, logical_OR, getMT, Delta_R, logical_AND_List*
#from pillar import *


print "starting clock"
start = time.clock()


debug_ = True

usage = "analyzer for t+DM (debugging) "
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-i", "--inputfile",  dest="inputfile")
parser.add_argument("-o", "--outputfile",
                    dest="outputfile", default="out.root")
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
https://lhcb.github.io/analysis-essentials/python/firhistogram.html
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
    print "running the code for ", infile_
    #tag_str = infile_.split("_13TeV")[0].split("Merged_")[1]

#    cross_section_weight = xsweight(tag_str)

#    print infile_, tag_str, cross_section_weight
    print infile_

    # dataframe for output
    df_out = DataFrame(columns=['run', 'lumi', 'event', 'MET', 'MT_ele','MT_mu', 'Njets_PassID', 'Nbjets_PassID', 'Ele1Pt', 'Ele1Eta', 'Ele1Phi','Ele2Pt', 'Ele2Eta', 'Ele2Phi', 'Mu1Pt', 'Mu1Eta', 'Mu1Phi','Mu2Pt', 'Mu2Eta', 'Mu2Phi','nTau','Jet1Pt','Jet1Eta', 'Jet1Phi', 'Jet2Pt', 'Jet2Eta', 'Jet2Phi', 'Jet3Pt', 'Jet3Eta', 'Jet3Phi', 'Jet1Idx', 'Jet2Idx', 'Jet3Idx'])

    recoil_den = TH1F("recoil_den", "recoil_den", 100, 0.0, 1000.)
    recoil_num = TH1F("recoil_num", "recoil_num", 100, 0.0, 1000.)

    jetvariables = ['THINnJet', 'THINjetPx', 'THINjetPy', 'THINjetPz', 'THINjetEnergy', 'THINjetCISVV2', 'THINjetHadronFlavor', 'THINjetNHadEF', 'THINjetCHadEF', 'THINjetCEmEF', 'THINjetPhoEF', 'THINjetEleEF', 'THINjetMuoEF', 'THINjetCorrUncUp', 'runId', 'lumiSection', 'eventId', 'pfMetCorrPt', 'pfMetCorrPhi', 'pfMetCorrUnc', 'isData','hlt_trigName','hlt_trigResult', 'nEle', 'elePx', 'elePy', 'elePz', 'eleEnergy', 'eleIsPassLoose', 'eleIsPassTight', 'nMu', 'muPx', 'muPy', 'muPz', 'muEnergy', 'isTightMuon','muChHadIso', 'muNeHadIso', 'muGamIso', 'muPUPt', 'muCharge', 'HPSTau_n','HPSTau_Px','HPSTau_Py','HPSTau_Pz','HPSTau_Energy','disc_decayModeFinding','disc_byLooseIsolationMVA3oldDMwLT']

    filename = infile_

    ''' global variables, mainly to be stored in the new root tree for quick analysis or histo saving '''

    icount = 0

    df_new = DataFrame()
    df_all = DataFrame()

    ieve = 0
    jetptseries = []
    jetetaseries = []
    jetphiseries = []
    jet_pt30 = []
    jet_pt50 = []
    jet_eta4p5 = []
    jet_IDtightVeto = []
    jet_eta2p4 = []
    jet_NJpt30 = []
    jet_NJpt30_Eta4p5 = []
    jet_csvmedium = []
    jet_N_bmedium_eta2p4_pt30 = []
    hlt_ele = []
    met_250_ = []
    for df in read_root(filename, columns=jetvariables, chunksize=125000):
        icount = icount + 1
        ''' all the operations which should be applied to each event must be done under this loop,
        otherwise effect will be reflected on the last chunk only. Each chunck can be considered
        as a small rootfile. An example of how to add new variable and copy the new dataframe into
        a bigger dataframe is shown below. This is the by far fastest method I manage to find, uproot
        awkward arrays are even faster but difficult to use on lxplus. and may be on condor.  '''
        for nak4jet_, ak4px_, ak4py_, ak4pz_, ak4e_, ak4csv, ak4flavor, ak4NHEF, ak4CHEF, ak4CEmEF, ak4PhEF, ek4EleEF, ak4MuEF, ak4JEC, trigName_,trigResult_, nele_, elepx_, elepy_, elepz_, elee_, elelooseid_, eletightid_, nmu_, mupx_, mupy_, mupz_, mue_, mutightid_, muChHadIso_, muNeHadIso_, muGamIso_, muPUPt_, muCharge_, met_, metphi_, run, lumi, event, nTau_, tau_px_, tau_py_, tau_pz_, tau_e_, tau_dm_, tau_isLoose_ in zip(df.THINnJet, df.THINjetPx, df.THINjetPy, df.THINjetPz, df.THINjetEnergy, df.THINjetCISVV2, df.THINjetHadronFlavor, df.THINjetNHadEF, df.THINjetCHadEF, df.THINjetCEmEF, df.THINjetPhoEF, df.THINjetEleEF, df.THINjetMuoEF, df.THINjetCorrUncUp, df.hlt_trigName, df.hlt_trigResult, df.nEle, df.elePx, df.elePy, df.elePz, df.eleEnergy, df.eleIsPassLoose, df.eleIsPassTight, df.nMu, df.muPx, df.muPy, df.muPz, df.muEnergy, df.isTightMuon, df.muChHadIso, df.muNeHadIso, df.muGamIso, df.muPUPt, df.muCharge, df.pfMetCorrPt, df.pfMetCorrPhi, df.runId, df.lumiSection, df.eventId, df.HPSTau_n,df.HPSTau_Px,df.HPSTau_Py,df.HPSTau_Pz,df.HPSTau_Energy, df.disc_decayModeFinding, df.disc_byLooseIsolationMVA3oldDMwLT):

            print "ievent = ", ieve

            ieve = ieve + 1
            if debug_:
                print nak4jet_, ak4px_, ak4py_, ak4pz_, ak4e_

            '''
            *******   *****   *******
               *      *          *
               *      ****       *
               *      *          *
            ***       *****      *
            '''

            ''' This small block compute the pt of the jet and add them back
            into the original dataframe as a next step for further usage. '''
            ak4pt = [getPt(ak4px_[ij], ak4py_[ij]) for ij in range(nak4jet_)]
            jetptseries.append(ak4pt)

            ''' Jet Loose ID is already applied in the preselection  '''

            ''' eta and phi of the ak4 jets '''
            ak4eta = [getEta(ak4px_[ij], ak4py_[ij], ak4pz_[ij])
                      for ij in range(nak4jet_)]
            ak4phi = [getPhi(ak4px_[ij], ak4py_[ij]) for ij in range(nak4jet_)]

            jetetaseries.append(ak4eta)
            jetphiseries.append(ak4phi)

            ''' pT>30 GeV, |eta|<4.5 is already applied in the tuples '''

            ''' jets with pt > 30 GeV '''
            ak4_pt30 = [(ak4pt[ij] > 30.) for ij in range(nak4jet_)]
            jet_pt30.append(ak4_pt30)

            ''' jets with pt > 50 GeV '''
            ak4_pt50 = [(ak4pt[ij] > 50.) for ij in range(nak4jet_)]
            jet_pt50.append(ak4_pt50)

            ''' jet |eta| < 4.5  '''
            ak4_eta4p5 = [(abs(ak4eta[ij]) < 4.5) for ij in range(nak4jet_)]
            jet_eta4p5.append(ak4_eta4p5)

            ''' jet |eta| < 2.4 '''
            ak4_eta2p4 = [(abs(ak4eta[ij]) < 2.4) for ij in range(nak4jet_)]
            jet_eta2p4.append(ak4_eta2p4)

            ''' jet tightLeptonVeto ID to reject fake jets coming from the leptons, Veto ID should be applied only for jets within the detector abs(eta) < 2.4

            Following the syntax of if else in list comprehension
            [f(x) if condition else g(x) for x in sequence]
            '''

            ak4_IDtightVeto = [((ak4NHEF[ij] < 0.90) and (ak4PhEF[ij] < 0.90) and (ak4MuEF[ij] < 0.8) and (ak4CEmEF[ij] < 0.90) and abs(ak4eta[ij]) < 2.4)
                               or ((ak4NHEF[ij] < 0.90) and (ak4PhEF[ij] < 0.90) and (ak4MuEF[ij] < 0.8) and abs(ak4eta[ij]) < 2.7 and abs(ak4eta[ij]) > 2.4) if (abs(ak4eta[ij]) < 2.7)
                               else True for ij in range(nak4jet_)]
            jet_IDtightVeto.append(ak4_IDtightVeto)

            if debug_:
                print "ak4_IDtightVeto", ak4_IDtightVeto

            ''' njets passing jet pt > 30 and eta < 4.5 and Loose Jet ID '''
            jet_NJpt30.append(ak4_pt30.count(True))

            jet_NJpt30_Eta4p5.append(
                len([ij for ij in range(nak4jet_) if (ak4_eta4p5[ij] and ak4_pt50[ij])]))

            ak4_csvmedium = [(ak4csv[ij] > 0.8484) for ij in range(nak4jet_)]
            jet_csvmedium.append(ak4_csvmedium)

            jet_N_bmedium_eta2p4_pt30.append(len([ij for ij in range(nak4jet_) if (
                (ak4_eta2p4[ij]) and (ak4_pt30[ij]) and (ak4_csvmedium[ij]))]))

            '''

            ****   *      ****
            *      *      *
            ***    *      ***
            *      *      *
            ****   ****   ****

            the selection for the electron is done here, later the new branches are added to the dataframe.

            '''

            ''' electron triggers '''

            #hlt_ele.append(logical_OR([hlt_ele27, hlt_ele105, hlt_ele115, hlt_ele32, hlt_ele32_eta2p1, hlt_ele27_eta2p1]))

            if debug_:
                print "event ------", event
            ''' get pt, eta, phi of electrons '''
            elept = [getPt(elepx_[ie], elepy_[ie]) for ie in range(nele_)]
            eleeta = [getEta(elepx_[ie], elepy_[ie], elepz_[ie])
                      for ie in range(nele_)]
            elephi = [getPhi(elepx_[ie], elepy_[ie]) for ie in range(nele_)]
            ''' electron pt and eta cut, tuples already have electron pT > 10 GeV and |eta|<2.5
            Veto electron ID is also applied on the electron at preselection level '''
            ele_pt10 = [(elept[ie] > 10) for ie in range(nele_)]
            ele_pt30 = [(elept[ie] > 30) for ie in range(nele_)]

            ele_IDLoose = [(elelooseid_[ie]) for ie in range(nele_)]
            ele_IDTight = [(eletightid_[ie]) for ie in range(nele_)]
            ele_eta2p1 = [(abs(eleeta[ie]) < 2.1) for ie in range(nele_)]
            ele_eta2p5 = [(abs(eleeta[ie]) < 2.5) for ie in range(nele_)]

            ele_pt10_eta2p5_vetoID = []
            if len(ele_pt10) > 0:
                ele_pt10_eta2p5_vetoID = logical_AND_List2(
                    ele_pt10, ele_eta2p5)

            if debug_:
                print "ele info"
            if debug_:
                print "pt, id eta =", ele_pt30, ele_IDTight, ele_eta2p1
            if debug_:
                for ie in range(nele_):
                    print elept[ie], eleeta[ie], eletightid_[
                        ie], elepx_[ie], elepy_[ie], elepz_[ie]

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
            mupt = [getPt(mupx_[imu], mupy_[imu]) for imu in range(nmu_)]
            mueta = [getEta(mupx_[imu], mupy_[imu], mupz_[imu])
                     for imu in range(nmu_)]
            muphi = [getPhi(mupx_[imu], mupy_[imu]) for imu in range(nmu_)]

            ''' For vetoing in the electron region only Looose Mu ID and ISo with pt > 10 GeV is needed and is already applied in the skimmer '''
            muIso_ = [((muChHadIso_[imu]+ max(0., muNeHadIso_[imu] + muGamIso_[imu] - 0.5*muPUPt_[imu]))/mupt[imu]) for imu in range(nmu_)]

            mu_pt10 = [(mupt[imu] > 10.0) for imu in range(nmu_)]
            mu_pt30 = [(mupt[imu] > 30.0) for imu in range(nmu_)]
            mu_eta2p4 = [(abs(mueta[imu]) < 2.4) for imu in range(nmu_)]
            mu_IDTight = [mutightid_[imu] for imu in range(nmu_)]
            mu_IsoTight = [(muIso_[imu] < 0.15) for imu in range(nmu_)]

            mu_pt10_eta2p4_looseID = []
            if len(mu_pt10):
                mu_pt10_eta2p4_looseID = logical_AND_List2(mu_pt10, mu_eta2p4)


            ''' MET SELECTION '''
            met_250_.append(met_ > 250.0)

            ''' MT Calculation for electrons '''
            mt_ele = [getMT(elept[ie], met_, elephi[ie], metphi_) for ie in range(nele_)]
            # mt_ele_.append(mt_ele)

            ''' MT Calculation for muons '''
            mt_mu = [getMT(mupt[imu], met_, muphi[imu], metphi_) for imu in range(nmu_)]
            # mt_mu_.append(mt_mu)

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
            ele_eta2p1_idT_pt30 = []
            if (len(ele_eta2p1) > 0):
                ele_eta2p1_idT_pt30 = logical_AND_List3(
                    ele_eta2p1, ele_IDTight, ele_pt30)

            '''
            > 0 means >= 1. The selection in the function is implemented like >= not >. Therefore pay attention when using this function.
            The function also take care of the fact that the operation will be performed only when size of the list is >= N, where N is by default 0 and has to be provided
            '''
            pass_ele_index = WhereIsTrue(ele_eta2p1_idT_pt30, 1)

            pass_veto_id_ele_index = WhereIsTrue(ele_pt10_eta2p5_vetoID, 1)

            mu_eta2p4_idT_pt30 = []
            if (len(mu_pt30) > 0):
                mu_eta2p4_idT_pt30 = logical_AND_List3(
                    mu_pt30, mu_IDTight, mu_IsoTight)

            pass_mu_index = WhereIsTrue(mu_eta2p4_idT_pt30, 1)

            ak4_pt30_eta4p5_IDL = []
            if len(ak4_pt30) > 0:
                ak4_pt30_eta4p5_IDL = logical_AND_List2(ak4_pt30, ak4_eta4p5)

            ''' we need at least 3 jets passing id, so we must ensure presene of 3 jets atleast '''
            pass_jet_index = WhereIsTrue(ak4_pt30_eta4p5_IDL, 3)

            '''

            ********    *        *       *
               *      *    *     *       *
               *     *      *    *       *
               *     ********    *       *
               *     *      *    *       *
               *     *      *     ******
            the selection for the tau is done here, later the new columns are added to the dataframe for each of them.

            '''

            ''' tau pt threshold and eta threshold, tuples already have tau pt > 10 and |eta| < 2.4 '''
            taupt = [getPt(tau_px_[itau], tau_py_[itau]) for itau in range(nTau_)]

            ''' eta and phi of the tau '''
            taueta = [getEta(tau_px_[itau], tau_py_[itau], tau_pz_[itau]) for itau in range(nTau_)]
            tauphi = [getPhi(tau_px_[itau], tau_py_[itau]) for itau in range(nTau_)]

            tau_pt18 = [(taupt[itau] > 18.0) for itau in range(nTau_)]
            tau_eta2p3 = [(abs(taueta[itau]) < 2.3) for itau in range(nTau_)]

            tau_IDLoose = [(tau_isLoose_[itau]) for itau in range(nTau_)]
            tau_DM = [(tau_dm_[itau]) for itau in range(nTau_)]
            tau_iDLdm = []
            tau_iDLdm = logical_AND_List2(tau_IDLoose,tau_DM)

            ''' take AND of all the tau cuts (just take the lists) '''
            tau_eta2p3_iDLdm_pt18 = []
            if (len(tau_eta2p3) > 0):
                tau_eta2p3_iDLdm_pt18 = logical_AND_List3(
                    tau_eta2p3, tau_iDLdm, tau_pt18 )

            pass_tau_index = WhereIsTrue(tau_eta2p3_iDLdm_pt18,1)

            '''

            All the object selection is done before this,
            region specific cuts are here.

            '''

            jetCleanAgainstEle = []
            for ijet in range(len(ak4_pt30_eta4p5_IDL)):
                pass_ijet_iele_ = []
                for iele in range(len(ele_pt10_eta2p5_vetoID)):
                    pass_ijet_iele_.append(ak4_pt30_eta4p5_IDL[ijet] & ele_pt10_eta2p5_vetoID[iele] & (
                        Delta_R(ak4eta[ijet], eleeta[iele], ak4phi[ijet], elephi[iele]) > 0.4))
                print "pass_ijet_iele_ = ", pass_ijet_iele_
                # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
                jetCleanAgainstEle.append(
                    len(WhereIsTrue(pass_ijet_iele_)) == len(pass_ijet_iele_))
                print "jetCleanAgainstEle = ", jetCleanAgainstEle

            jetCleanAgainstMu = []
            for ijet in range(len(ak4_pt30_eta4p5_IDL)):
                pass_ijet_imu_ = []
                for imu in range(len(mu_pt10_eta2p4_looseID)):
                    pass_ijet_imu_.append(ak4_pt30_eta4p5_IDL[ijet] & mu_pt10_eta2p4_looseID[imu] & (
                        Delta_R(ak4eta[ijet], mueta[imu], ak4phi[ijet], muphi[imu]) > 0.4))
                # if the number of true is equal to length of vector then it is ok to keep this jet, otherwise this is not cleaned
                print "pass_ijet_imu_ = ", pass_ijet_imu_
                jetCleanAgainstMu.append(
                    len(WhereIsTrue(pass_ijet_imu_)) == len(pass_ijet_imu_))
                print "jetCleanAgainstMu = ", jetCleanAgainstMu
            jetCleaned = logical_AND_List2(
                jetCleanAgainstEle, jetCleanAgainstMu)
            print "jetCleaned = ", jetCleaned

            print "nele, nmu = ", ele_pt10_eta2p5_vetoID, mu_pt10_eta2p4_looseID

            pass_jet_index_cleaned = []
            pass_jet_index_cleaned = WhereIsTrue(jetCleaned, 3)

            print "pass_jet_index_cleaned = ", pass_jet_index_cleaned

            ak4_bjetM_eta2p4 = []
            if len(ak4_csvmedium) > 0:
                ak4_bjetM_eta2p4 = logical_AND_List3(
                    ak4_csvmedium, ak4_eta2p4, jetCleaned)
            pass_bjetM_eta2p4_index = WhereIsTrue(ak4_bjetM_eta2p4, 1)

            j1idx   = -1; j2idx   = -1; j3idx   = -1
            ele1idx = -1; ele2idx = -1
            mu1idx  = -1; mu2idx  = -1
            ele1idx = pass_ele_index[0]; ele2idx = pass_ele_index[1]
            mu1idx = pass_mu_index[0]; mu2idx = pass_mu_index[1]
            j1idx = pass_jet_index_cleaned[0]
            j2idx = pass_jet_index_cleaned[1]
            j3idx = pass_jet_index_cleaned[2]
            print 'reached here1'
            df_out = df_out.append({'run':run, 'lumi':lumi, 'event':event,
                                    'MET': met_, 'MT_ele': mt_ele[pass_ele_index[0]],'MT_mu': mt_mu[pass_mu_index[0]], 'Njets_PassID': len(pass_jet_index_cleaned), 'Nbjets_PassID':len(pass_bjetM_eta2p4_index),
                                    'Ele1Pt':elept[ele1idx], 'Ele1Eta':eleeta[ele1idx], 'Ele1Phi':elephi[ele1idx],'Ele2Pt':elept[ele2idx], 'Ele2Eta':eleeta[ele2idx], 'Ele2Phi':elephi[ele2idx],'Mu1Pt':mupt[mu1idx], 'Mu1Eta':mueta[mu1idx],
                                    'Mu1Phi':muphi[mu1idx],'Mu2Pt':mupt[mu2idx], 'Mu2Eta':mueta[mu2idx],'Mu2Phi':muphi[mu2idx],'nTau': len(pass_tau_index), 'Jet1Pt':ak4pt[j1idx], 'Jet1Eta':ak4eta[j1idx], 'Jet1Phi':ak4phi[j1idx],
                                    'Jet2Pt':ak4pt[j2idx],'Jet2Eta':ak4eta[j2idx], 'Jet2Phi':ak4phi[j2idx], 'Jet3Pt':ak4pt[j3idx],'Jet3Eta':ak4eta[j3idx],'Jet3Phi':ak4phi[j3idx],
                                    'Jet1Idx':j1idx, 'Jet2Idx':j2idx,'Jet3Idx':j3idx }, ignore_index=True)
            print 'reached here2'
            if debug_:
                print "object info", run, lumi, event, eleidx, elept[eleidx], eleeta[eleidx], elephi[eleidx], j1idx, ak4pt[j1idx], ak4eta[j1idx], ak4phi[j1idx], j2idx, ak4pt[j2idx], ak4eta[j2idx], ak4phi[j2idx], j3idx, ak4pt[
                    j3idx], ak4eta[j3idx], ak4phi[j3idx], met_, mt_ele[pass_ele_index[0]], [len(pass_ele_index) == 1, nmu_ == 0, met_ > 250.0, len(pass_jet_index) >= 3, len(pass_bjetM_eta2p4_index) == 0, mt_ele[pass_ele_index[0]] < 160.]

        df_all = concat([df_all, df])

    if debug_:
        print df_out

    outputfilename = args.outputfile
    df_out.to_root(outputfilename, key='bb_dm')

    # df_out_wmunu_cr.to_root(outputfilename, key='t_dm_wmunucr', mode='a')

    end = time.clock()
    print "%.4gs" % (end - start)

#files=['/tmp/khurana/ExoPieInput_tDM_06052019/Merged_WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_MC25ns_LegacyMC_20170328_0000_1.root', '/tmp/khurana/ExoPieInput_tDM_06052019/Merged_WJetsToLNu_HT-400To600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_MC25ns_LegacyMC_20170328_0000_2.root']


files = ['NCUGlobalTuples.root']
if __name__ == '__main__':
    try:
        pool = mp.Pool(1)
        pool.map(runtdm, files)
        pool.close()
    except Exception as e:
        print traceback.format_exc()
        pass
