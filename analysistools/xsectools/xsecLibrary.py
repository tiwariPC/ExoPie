import os, sys
#//--------------------------------------------------------------------------------------
def getXsec(samplename):
    Xsec = 1.0
    qcd_sf = 1.0
    #zKF = 1.23
    zKF = 1.0
    #wKF = 1.21
    wKF = 1.0
    #print (samplename)
    xs={'DYJetsToLL_M-50_HT-70to100_TuneCUETP8M1'    : 169.9 * zKF,
        'DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1'   : 147.4 * zKF,
        'DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1'   : 40.99 * zKF,
        'DYJetsToLL_M-50_HT-400to600_TuneCUETP8M1'   : 5.678 * zKF,
        'DYJetsToLL_M-50_HT-600to800_TuneCUETP8M1'   : 1.367 * zKF,
        'DYJetsToLL_M-50_HT-800to1200_TuneCUETP8M1'  : 0.6304 * zKF,
        'DYJetsToLL_M-50_HT-1200to2500_TuneCUETP8M1' : 0.1514 * zKF,
        'DYJetsToLL_M-50_HT-2500toInf_TuneCUETP8M1'  : 0.003565 * zKF,
        'ZJetsToNuNu_HT-100To200'       : 280.35 * zKF,
        'ZJetsToNuNu_HT-200To400'       : 77.67 * zKF,
        'ZJetsToNuNu_HT-400To600'       : 10.73 * zKF,
        'ZJetsToNuNu_HT-600To800'       : 2.559 * zKF,
        'ZJetsToNuNu_HT-800To1200'      : 1.1796 * zKF,
        'ZJetsToNuNu_HT-1200To2500'     : 0.28833 * zKF,
        'ZJetsToNuNu_HT-2500ToInf'      : 0.006945 * zKF,
        'WJetsToLNu_HT-70To100_TuneCUETP8M1'   : 1372.0 * wKF,
        'WJetsToLNu_HT-100To200_TuneCUETP8M1'  : 1345.0 * wKF,
        'WJetsToLNu_HT-200To400_TuneCUETP8M1'  : 359.7 * wKF,
        'WJetsToLNu_HT-400To600_TuneCUETP8M1'  : 48.91 * wKF,
        'WJetsToLNu_HT-600To800_TuneCUETP8M1'  : 12.05 * wKF,
        'WJetsToLNu_HT-800To1200_TuneCUETP8M1' : 5.501 * wKF,
        'WJetsToLNu_HT-1200To2500_TuneCUETP8M1': 1.329 * wKF,
        'WJetsToLNu_HT-2500ToInf_TuneCUETP8M1' : 0.03216 * wKF,
        'GJets_HT-40To100_TuneCUETP8M1' : 20790,
        'GJets_HT-100To200_TuneCUETP8M1': 9238,
        'GJets_HT-200To400_TuneCUETP8M1': 2305,
        'GJets_HT-400To600_TuneCUETP8M1': 274.4,
        'GJets_HT-600ToInf_TuneCUETP8M1': 93.46,
        'QCD_HT500to700_TuneCUETP8M1'   : 32100 * qcd_sf,
        'QCD_HT700to1000_TuneCUETP8M1'  : 6831 * qcd_sf,
        'QCD_HT1000to1500_TuneCUETP8M1' : 1207 * qcd_sf,
        'QCD_HT1500to2000_TuneCUETP8M1' : 119.9 * qcd_sf,
        'QCD_HT2000toInf_TuneCUETP8M1'  : 25.24 * qcd_sf,
        'TT_TuneCUETP8M2T4'             : 831.76,
        'WWTo1L1Nu2Q'                   : 49.997,
        'WWTo2L2Nu'                     : 12.178,
        'WWTo4Q_4f'                     : 51.723,
        'WZTo1L1Nu2Q'                   : 10.71,
        'WZTo1L3Nu'                     : 3.0330,
        'WZTo2L2Q'                      : 5.5950,
        'WZTo2Q2Nu'                     : 6.4880,
        'WZTo3LNu_TuneCUETP8M1'         : 4.4297,
        'ZZTo2L2Q'                      : 3.22,
        'ZZTo2Q2Nu'                     : 4.04,
        'ZZTo4L'                        : 1.2120,
        'ZZTo4Q'                        : 6.842,
        'TTWJetsToLNu_TuneCUETP8M1'     : 0.2043,
        'TTWJetsToQQ_TuneCUETP8M1'      : 0.4062,
        'TTZToQQ_TuneCUETP8M1'          : 0.5297,
        'ST_s-channel_4f_leptonDecays'   : 3.36,
        'ST_t-channel_top_4f_inclusiveDecays':136.02,
        'ST_t-channel_antitop_4f_inclusiveDecays':80.95, ## may be same as above, in that case change the above str
        'ST_tW_antitop_5f_inclusiveDecays':35.85,  ## ## may be same as above, in that case change the above str
        'TTToSemilepton_TuneCUETP8M2_ttHtranche3':364.35,
        'TTZToLLNuNu_M-10_TuneCUETP8M1':0.2529,  ## ## may be same as above, in that case change the above str
        'TTTo2L2Nu_TuneCUETP8M2_ttHtranche3':87.31, ## ## may be same as above, in that case change the above str
        'ST_tW_top_5f_inclusiveDecays':35.85,  ## ## may be same as above, in that case change the above str
        'WminusH_HToBB_WToLNu_M125'     : 0.100,
        'WplusH_HToBB_WToLNu_M125':0.159,
        'ggZH_HToBB_ZToLL_M125':0.007842,
        'ZH_HToBB_ZToLL_M125':0.04865,
        'WW_TuneCUETP8M1':118.7,
        'WZ_TuneCUETP8M1':47.2,
        'ZZ_TuneCUETP8M1':16.6,
        'SingleElectron-Run2016.root':1, ## the cross-section will always remain 1 here but the str should change
        'SingleMu2016.root':1 ## the cross-section will always remain 1 here but the str should change

    }
    return xs[samplename]
