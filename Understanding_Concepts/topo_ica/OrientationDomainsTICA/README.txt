
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                          %
%   Topographic Independent Component Analysis reveals     % 
%    random scrambling of orientation in visual space      %
%                                                          %
%   by M. Martinez-Garcia, L. M. Martinez and J. Malo      %
%                   (PLOS ONE, 2017)                       %
%  -----------------------------------------------------   %

http://isp.uv.es/code/visioncolor/TICAdomains/TICAdomains.html

%  -----------------------------------------------------   %
%                                                          %
% CODE AND DATA by Marina Martinez Garcia and Jesus Malo   % 
% to reproduce the results in the paper:                   %
%                                                          %
%    0 Domains in ferret (specific script & data)          %
%    1 Gather image data (specific scripts & data)         %
%    2 Extended Topographic ICA (general purpose code)     %
%    3 Analysis of Orientation Domains (specific scripts)  %
%                                                          %                                                         
% The new analysis of Topographic ICA presented here       %
% requires the followig publicly available toolboxes       %
%  (1) ImageICA by Hoyer and Hyvarinen,                    %
%      https://research.ics.aalto.fi/ica/imageica/         %
%  (2) BasicVideoTools by Malo et al.,                     %
%      http://isp.uv.es/code/visioncolor/Basic_Video.html  %
% These are included in this package for your convenience. %
% However, please also acknowledge the authors of the      %
% above toolboxes if you use our code.                     %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0. DOMAINS IN FERRET (actual biological data, not related to the computational 
%          goal of the paper, here just for fun):
%  - Script to visualize experimental orientation domains 
%    (intrinsic imaging in ferret)
%  - Folder: 0_Domains_in_ferret

jesus_test.m  
    % Loads some example data (http://isp.uv.es/code/visioncolor/TICAdomains/paper/example_ferret_data.zip) 
    % normalizes the images and represents nice temporal slices.
    % Paths and (more importantly) key frames and regions of interest are hardcoded 
    % i.e. [they were preselected by looking at all of them and selecting by hand] 
    % Requires    norm_image.m
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. IMAGE STATISTICS DATA: 
%  - Script to gather data at different resolutions (pictures come from the imageica toolbox)
%  - Script to compute the PCA complexity of the datasets
%  - Folder: 1_Gather_data

generate_data_tica_no_aliasing.m 
   % This script generates images, controlling visual angle and resolution. 
   % These images are the trining data of the different TICA configurations 
   % Warning! paths are hardcoded

pca_dimensionality.m
   % Computes the PCA dimensionality that captures 99% of the signal energy
   % Warning! paths harcoded
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. EXTENDED TOPOGRAPHIC ICA: 
%  - Convenient extension of Hyvarinen&Hoyer TICA code to tune nonlinearities, neighborhoods and starting point)
%  - Folder: 2_ExtendedImageICA
% 

estimate_MMG_JM.m    
   % Generalized estimation of the TICA basis (nonlinearities, neighborhoods and starting point)
   % It includes GenerateNB_MMG_JM.m for more general neighborhoods.   
   % It has extra parameters. See the help and example on how to use it here: 
   % orientation_maps_ticas_MMG_JM.m  
   % Equivalent examples with classical estimate here: orientation_maps_ticas_I_J.m
   % Example scripts use whiten_hyva.m   % Where Whitening is not removing the mean but the mean luminance from each block
   % non_linearities.m % Plots examples of the functions used at the nonlinear part of the fast ICAs,
             	       % to understand why we choosed those particular parameters
   % Inicialitzar_2N_apartir_N.m % It crates a starting point for the TICA with 
                                 % size 2*N given a TICA basis of size N

ica_mich.m 
   % Efficient ICA implementation by Michael Gutmann

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. ANALYSIS:  
%  - Fit TICA functions to Gabor functions for parameter analysis (location and freq.)  
%    (Gabor functions sampled with controlled resolution require BasicVideoTools)
%  - Presentation of results in the spatial and frequency domains
%  - Check the presence of clusters in the sensors tuned to certain orientation (stat test)
%  - Folder: 3_Analysis   

figs_seccion3.m 
     % Script to reproduces the main result of the paper.
     % Explains procedure & gathers results after making the fits (Warning with paths!).
     % It requires calling these functions:
             data_analysis_others.m
             sort_basis.m    % sorts the column vectors of matrix M according 
                             % to their 2D spatio-frequency meaning (fitting of gabor functions) 
                   fit_gabor_f_phase_x.m
                   fit_2_gaussians.m     
                   fit_gaussian.m        
                   fit_gabor.m
                   gaussiana_2d.m  

              analysis_others.m  % makes plots
     % Special case N=100 (shown in the web). 
     % Special case because lack of convergence forced us to select the Gabor-like functions by hand
         analysis_100.m
         analysis_100_manual.m
         analysis_100_sortu.m

     % Note (for internal record)
     % The functions actually used for the results (paper/web) 
     % were the ones with the first (non-optimized) modifications of estimate.m. See:
             figs_seccion3.m
             figs_H_3.m
             figs_nolin_3.m  

Test_salt_peper.m  
     % This script reproduces statistical test to reject non-uniformity, 
     % it does the simulated distributions and computes the test simulated vs real.
     % It makes the plot on the figure 7  

Fig_6_test_boost.m 
     % Script to make the plots on the figure 6.

Mdist3.m           % Computes a distance between two sets of 2-D points
 	'KL':            Kullback–Leibler divergence
	'bhattacharyya': Bhattacharyya distance (not used at the paper)
	'empty_cells':   Number of empty cells  (not used at the paper)
         bhattacharyya.m     Computes the bhattacharyya distance
         KLDiv.m             Kullback–Leibler divergence


save2pdf.m          % save a pdf figure cropping it properly (by Gabe Hoffmann)
Angle_color_bar.m   % Makes the color bar that indicates orientation


