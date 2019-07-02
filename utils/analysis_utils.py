import os
def getGenPt(sample,nGenPar, genParId, genMomParId, genParSt,genParP4):
    pt=0.0
    #################
    # WJets
    #################
    if sample=="WJETS":
    #if True:
        goodLepID = []
        for ig in range(nGenPar):
            PID    = genParId[ig]
            momPID = genMomParId[ig]
            status = genParSt[ig]
            if ( (abs(PID) != 11) & (abs(PID) != 12) &  (abs(PID) != 13) & (abs(PID) != 14) &  (abs(PID) != 15) &  (abs(PID) != 16) ): continue
            if ( ( (status != 1) & (abs(PID) != 15)) | ( (status != 2) & (abs(PID) == 15)) ): continue
            if ( (abs(momPID) != 24) & (momPID != PID) ): continue
            goodLepID.append(ig)

        if len(goodLepID) == 2 :
            l4_thisLep = genParP4[goodLepID[0]]
            l4_thatLep = genParP4[goodLepID[1]]
            l4_z = l4_thisLep + l4_thatLep
            pt = l4_z.Pt()

    #################
    #ZJets
    #################
    if sample == "ZJETS":
    #if True:
        goodLepID = []
        for ig in range(nGenPar):
            PID    = genParId[ig]
            momPID = genMomParId[ig]
            status = genParSt[ig]


            if ( (abs(PID) != 12) &  (abs(PID) != 14) &  (abs(PID) != 16) ) : continue
            if ( status != 1 ) : continue
            if ( (momPID != 23) & (momPID != PID) ) : continue
            goodLepID.append(ig)

        if len(goodLepID) == 2 :
            l4_thisLep = genParP4[goodLepID[0]]
            l4_thatLep = genParP4[goodLepID[1]]
            l4_z = l4_thisLep + l4_thatLep
            pt = l4_z.Pt()

    return pt


def GetTTPt(sample, nGenPar, genParId, genMomParId, genParSt, genParP4):
    #################
    #TTBar
    #################
    ptList=[]
    if (sample=="TT"):
        goodLepID = []
        for ig in range(nGenPar):
            PID    = genParId[ig]
            momPID = genMomParId[ig]
            status = genParSt[ig]
            if ( abs(PID) == 6) :
                goodLepID.append(ig)
        if(len(goodLepID)==2):
            l4_thisLep = genParP4[goodLepID[0]]
            l4_thatLep = genParP4[goodLepID[1]]
            ptList =  [l4_thisLep.Pt(), l4_thatLep.Pt()]
            
    return ptList
