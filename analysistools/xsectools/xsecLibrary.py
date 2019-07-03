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
    xs={'DYJetsToLL_M-50_HT-70to100'    : 169.9 * zKF,
        'DYJetsToLL_M-50_HT-100to200'   : 147.4 * zKF,
        'DYJetsToLL_M-50_HT-200to400'   : 40.99 * zKF,
        'DYJetsToLL_M-50_HT-400to600'   : 5.678 * zKF,
        'DYJetsToLL_M-50_HT-600to800'   : 1.367 * zKF,
        'DYJetsToLL_M-50_HT-800to1200'  : 0.6304 * zKF,
        'DYJetsToLL_M-50_HT-1200to2500' : 0.1514 * zKF,
        'DYJetsToLL_M-50_HT-2500toInf'  : 0.003565 * zKF,
        'ZJetsToNuNu_HT-100To200'       : 280.35 * zKF,
        'ZJetsToNuNu_HT-200To400'       : 77.67 * zKF,
        'ZJetsToNuNu_HT-400To600'       : 10.73 * zKF,
        'ZJetsToNuNu_HT-600To800'       : 2.559 * zKF,
        'ZJetsToNuNu_HT-800To1200'      : 1.1796 * zKF,
        'ZJetsToNuNu_HT-1200To2500'     : 0.28833 * zKF,
        'ZJetsToNuNu_HT-2500ToInf'      : 0.006945 * zKF,
        'WJetsToLNu_HT-70To100'         : 1372.0 * wKF,
        'WJetsToLNu_HT-100To200'        : 1345.0 * wKF,
        'WJetsToLNu_HT-200To400'        : 359.7 * wKF,
        'WJetsToLNu_HT-400To600'        : 48.91 * wKF,
        'WJetsToLNu_HT-600To800'        : 12.05 * wKF,
        'WJetsToLNu_HT-800To1200'       : 5.501 * wKF,
        'WJetsToLNu_HT-1200To2500'      : 1.329 * wKF,
        'WJetsToLNu_HT-2500ToInf'       : 0.03216 * wKF,
        'GJets_HT-40To100'              : 20790,
        'GJets_HT-100To200'             : 9238,
        'GJets_HT-200To400'             : 2305,
        'GJets_HT-400To600'             : 274.4,
        'GJets_HT-600ToInf'             : 93.46,
        'QCD_HT500to700'                : 32100 * qcd_sf,
        'QCD_HT700to1000'               : 6831 * qcd_sf,
        'QCD_HT1000to1500'              : 1207 * qcd_sf,
        'QCD_HT1500to2000'              : 119.9 * qcd_sf,
        'QCD_HT2000toInf'               : 25.24 * qcd_sf,
        'TT_TuneCUETP8M2T4'             : 831.76,
        'TTToSemilepton'                : 364.35,
        'TTTo2L2Nu'                     : 87.31,
        'WWTo1L1Nu2Q'                   : 49.997,
        'WWTo2L2Nu'                     : 12.178,
        'WWTo4Q_4f'                     : 51.723,
        'WZTo1L1Nu2Q'                   : 10.71,
        'WZTo1L3Nu'                     : 3.0330,
        'WZTo2L2Q'                      : 5.5950,
        'WZTo2Q2Nu'                     : 6.4880,
        'WZTo3LNu'                      : 4.4297,
        'ZZTo2L2Q'                      : 3.22,
        'ZZTo2Q2Nu'                     : 4.04,
        'ZZTo4L'                        : 1.2120,
        'ZZTo4Q'                        : 6.842,
        'ST_s-channel_4f_leptonDecay'   : 3.36,
        'ST_t-channel_antitop_4f'       : 80.95,
        'ST_t-channel_top_4f'           : 136.02,
        'ST_tW_antitop_5f'              : 35.85,
        'ST_tW_top_5f'                  : 35.85,
        'TTWJetsToLNu'                  : 0.2043,
        'TTWJetsToQQ'                   : 0.4062,
        'TTZToQQ'                       : 0.5297,
        'TTZToLLNuNu'                   : 0.2529
    }
    return xs[samplename]

