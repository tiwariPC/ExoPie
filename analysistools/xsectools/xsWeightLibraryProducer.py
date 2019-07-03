from os import listdir
from os.path import isfile, join

from ROOT import TFile, TH1F

from xsecLibrary  import getXsec

inputfilepath="/tmp/khurana/Merged_tDM_06052019/"

year_ = "2016"

lumi = 0.0
if year_=="2016": lumi = 36000
if year_=="2017": lumi = 41200
if year_=="2018": lumi = 60100



def FindIntegral(filename):
    infile_ = TFile(filename,'READ')
    h_total = TH1F()
    histname = 'h_total_mcweight'
    h_total = infile_.Get(histname)
    print filename,  h_total.Integral()
    #print filename, h_total.Integral()
    return h_total.Integral()


filenameList = listdir(inputfilepath)

FullfileList = [join(inputfilepath,ifile) for ifile in filenameList]
print filenameList
print FullfileList

integralList=[FindIntegral(ifile) for ifile in FullfileList]
print integralList

tag_str = [ifile.split("_13TeV")[0].split("Merged_")[1] for ifile in filenameList]

print tag_str


crosssection = [getXsec(itag) for itag in tag_str]
print crosssection

legend = [itag.split("_")[0] for itag in tag_str]


print "tag_str rootfileName total_events xsec xsec_weight Legend"# LineColor LineWidth LineStyle FillColor FillStyle"
#printall=[]
fout = open('crosssectionTable.txt','w')
for i in range(len(filenameList)):
    to_print = tag_str[i] + " " + filenameList[i] + " " + str(integralList[i]) + " " + str(crosssection[i]) + " " + str(crosssection[i]*lumi/integralList[i]) + " " + legend[i]# + " " + str(1) + " " + str(1) + " " + str(1) + " " + str(1) + " " + str(1)
    print >> fout, to_print
    #printall.append(to_print)
    #printall.append("\n")
    

#fout.write(printall)
fout.close()
