function data_analysis_others(file_in,file_out,fs)

%
%
% 
%  Mss                       2500x1600            32000000  double              
%  delt_xf_ang_phase1_t      1600x4                  51200  double              
%  delt_xf_ang_phase2_t      1600x4                  51200  double              
%  delt_xf_ang_phasem_t      1600x4                  51200  double              
%  er1_t                        1x1600               12800  double              
%  er2_t                        1x1600               12800  double              
%  erm_t                        1x1600               12800  double              
%  xf1_t                     1600x4                  51200  double              
%  xf2_t                     1600x4                  51200  double              
%  xfm_t                     1600x4                  51200  double         

load(file_in)
clear W

[Mss,xf1_t,delt_xf_ang_phase1_t,xf2_t,delt_xf_ang_phase2_t,xfm_t,delt_xf_ang_phasem_t,er1_t,er2_t,erm_t] = sort_basis(A,fs);
save(file_out)
