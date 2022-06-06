#include "TTree.h"
#include "TClass.h" 
#include "TH1.h" 

#include "TFile.h"
#include "TNtuple.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TTimeStamp.h"
#include <TParameter.h>
#include "TSystem.h"
#include "TSystemFile.h"
#include "TSystemDirectory.h"

#include "TGraph.h"

#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#include <stdio.h>
#include <string.h>

#include <sstream>
#include <cstdlib> 

void root_to_txt() {

  //Read root file -------------------------------------------------------------------------//
  //TString str_fin=Form("wf_4696_358_1_1_261.root");
  //TString str_fin=Form("wf_4696_358_1_0_188.root");
  //TString str_fin=Form("wf_4696_358_8_0_254.root");
  TString str_fin=Form("wf_4696_1_0_109.root");
  TString str_fout=str_fin + ".txt";
  TFile *fin = new TFile(str_fin);
  TH1F *H = (TH1F *)fin->Get("fRAWTQHisto");
  //H->Print();

  Int_t n_H = H->GetSize();
  cout << "n_H:" << n_H <<endl; 
  cout << str_fin <<endl; 
  cout << str_fout <<endl; 

  fstream fout; fout.open(str_fout.Data(),ios::out);

  for (int i=0; i<n_H; i++) {
      double cc=H->GetBinContent(i);
      //cout << H->GetBinContent(i) << endl;
      fout<<cc<<endl;
  }
  fout.close();


}
