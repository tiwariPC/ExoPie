void PreparePUWeights(){
  TFile* f = new TFile("data/pileUPinfo2016.root","READ");
  f->cd();
  
  TH1F* h = (TH1F*) f->Get("hpileUPhist");
  int nbins = h->GetNbinsX();
  
  std::cout<<"#include <iostream> "<<std::endl;
  std::cout<<"#include <fstream> "<<std::endl;
  ofstream f1;
  f1.open("PileUpWeights.h");
  
  f1<<"#ifndef PileUpWeights_h_"<<std::endl;
  f1<<"#define PileUpWeights_h_ "<<std::endl;
  f1<<"using namespace std; "<<std::endl;
  f1<<"class PileUpWeights{ "<<std::endl;
  f1<<" public: "<<std::endl;
  f1<<"   PileUpWeights(){};"<<std::endl;
  f1<<"   ~PileUpWeights(){};"<<std::endl;
  f1<<"   static Float_t PUWEIGHT(Int_t nvtx){"<<std::endl;
  f1<<"   Float_t  puweight[60];"<<std::endl;
  for (int i=1; i<=nbins; i++){
    //f1<< "   puweight["<<i-1<<"]  =  "<<h->GetBinContent(i)<<";"<<std::endl;
    f1<< "   puweight.append("<<h->GetBinContent(i)<<")"<<std::endl;
  }
  f1<<"   return puweight[nvtx];"<<std::endl;
  f1<<"  }"<<std::endl;
  f1<<"};"<<std::endl;
  f1<<"#endif"<<std::endl;
}
