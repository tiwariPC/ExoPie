# Exopie
bunch of scripts and macros to perform Exotica analysis using python 

## Installation 
cmsrel CMSSW_10_3_0
cd CMSSW_10_3_0/src
cmsenv
cd Exopie
git clone git@github.com:ramankhurana/ExoPie.git 
cd 

## Dependencies 
no CMS packages is used but cmsenv is good to get all the python libraries and tools, no extra package should be needed to run the setup. 

## Merging the input files
The script for merging is located in analysistools/mergetools 
It is better to merge the input files and use the chunksize of 100K to 150K (during the analysis) for better performances. 
The merging of the skimmed files can be done using: 

python merge_data_mc.py -i input/path/to/dir  -o output/path/to/dir -N Number_of_files_to_merge_in_one_file

by default it take all the files for a given sample in the directory structure provided and merge them into 1.  

## Run the t+DM analysis code (cross-check only) 
The analyzers files are located in analyzers/ and can be run using 
python tdmAnalyzer.py -i input_file -o output_file_name

if no outfile name is provided, default, out.root will be used. 

## Run the bb+MET analsis code,  developing, not yet on the git. 
python bbdmAnalyzer.py -i input_file -o output_file_name

## 