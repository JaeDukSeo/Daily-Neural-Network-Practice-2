
% The data plotted here come from: (1) TICA experiments, and (2) Analysis of functions
%
% (1) Different realizations of TICA over different image-samples from the imageICA
%     database assuming both different visual angle and different sampling frequency
%     (thus giving rise to different discrete image sizes).
%     
%     * Training sets generation: generate_data_tica.m
%                                 -> data_Fx_S_im_I.mat   x
%                                         F = 1,2,4 (sampling frequency)  
%                                         S = 16,20,32,50,100 (block sizes)
%                                         I = 1...13 (image) 
%
%     * TICA computation: orientation_maps_ticas_I_J.m (where i stands for "block size" and j stands for "sampling frequency")
%                         estimate.m
%                         -> tica_1x_16_image.m     
%                            tica_1x_20_image.m 
%                            tica_1x_32_image.m 
%                            tica_1x_50_image.m 
%                            tica_1x_100_image.m    (did not fully converge!)
%                            tica_2x_32_image.m 
%                            tica_4x_32_image.m 
%
%  (2) Analysis of TICA functions involves fitting the features to Gabors (with appropriate sampling frequency)
%        
%      * Fitting the functions from tica* files and saving the results:
%        (this can be done for the cases that fully converged) 
%                         data_analysis_others.m (uses sort_basis.m and associated functions)
%
%                         -> sortu_1x_16.mat  (fs = 25, xm = 0.64) 
%                         -> sortu_1x_20.mat  (fs = 25, xm = 0.8) 
%                         -> sortu_1x_32.mat  (fs = 25, xm = 1.28) 
%                         -> sortu_1x_50.mat  (fs = 25, xm = 2) 
%                         -> sortu_2x_32.mat  (fs = 12.5, xm = 2.56) 
%                         -> sortu_4x_32.mat  (fs = 6.25, xm = 5.12)
%
%      * The 100 case had to be analyzed by hand since convergence comes in
%        junks in separate locations of the topology.
%        We isolated those regions by hand and prepared them for the analysis.
%
%        - Identification of the regions was done in: analysis_100_sortu.m 
%                         -> sortu_1x_100_1.mat  (fs = 25, xm = 4)
%                         -> sortu_1x_100_2.mat 
%                         -> sortu_1x_100_3.mat 
%                         -> sortu_1x_100_4.mat 
%
%        Appropriate error threshold was set by hand so that not many noisy functions were considered.  
%

%%%%%%%%%%%%%%%%%%%%%%
%  RESULTADOS 16 - 50
%%%%%%%%%%%%%%%%%%%%%%

file = 'sortu_1x_16';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 0.64;
b = 2;
umbral_error = 0.18;
size_dots_x = 5;
size_dots_f = 11;
size_dots_x2 = 140/4;
fig = 1;
[xy_buenas064,frec_buenas064,ind_buenas,colores_map_rgb,error16,A,Ap16,Ap_net,Afit16] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);
 
file = 'sortu_1x_16_nl_2';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 0.64;
b = 2;
umbral_error = 0.25;
size_dots_x = 5;
size_dots_f = 11;
size_dots_x2 = 140/4;
fig = 10;
[xy_buenas0642,frec_buenas0642,ind_buenas,colores_map_rgb,error16,A,Ap16,Ap_net,Afit16] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);
 
file = 'sortu_1x_16_nl_3';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 0.64;
b = 2;
umbral_error = 0.25;
size_dots_x = 5;
size_dots_f = 11;
size_dots_x2 = 140/4;
fig = 20;
[xy_buenas0643,frec_buenas0643,ind_buenas,colores_map_rgb,error16,A,Ap16,Ap_net,Afit16] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);
 
file = 'sortu_1x_16_nl_4';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 0.64;
b = 2;
umbral_error = 0.25;
size_dots_x = 5;
size_dots_f = 11;
size_dots_x2 = 140/4;
fig = 30;
[xy_buenas0644,frec_buenas0644,ind_buenas,colores_map_rgb,error16,A,Ap16,Ap_net,Afit16] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);

%%%%%%%%%%%%%%%%

file = 'sortu_1x_32';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 1.28;
b = 3;
umbral_error = 0.18;
size_dots_x = 6;
size_dots_f = 11;
size_dots_x2 = (0.64/1.28)*140/3;
fig = 40;
[xy_buenas128,frec_buenas128,ind_buenas,colores_map_rgb,error32,A,Ap32,Ap_net,Afit32] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);

file = 'sortu_1x_32_nl_2';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 1.28;
b = 3;
umbral_error = 0.18;
size_dots_x = 6;
size_dots_f = 11;
size_dots_x2 = (0.64/1.28)*140/3;
fig = 50;
[xy_buenas1282,frec_buenas1282,ind_buenas,colores_map_rgb,error32,A,Ap32,Ap_net,Afit32] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);

file = 'sortu_1x_32_nl_3';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 1.28;
b = 3;
umbral_error = 0.18;
size_dots_x = 6;
size_dots_f = 11;
size_dots_x2 = (0.64/1.28)*140/3;
fig = 60;
[xy_buenas1283,frec_buenas1283,ind_buenas,colores_map_rgb,error32,A,Ap32,Ap_net,Afit32] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);

file = 'sortu_1x_32_nl_4';
fsx = 25;
xmax = 5.12;
f_lim = 9;
x_lim = 1.28;
b = 3;
umbral_error = 0.18;
size_dots_x = 6;
size_dots_f = 11;
size_dots_x2 = (0.64/1.28)*140/3;
fig = 70;
[xy_buenas1284,frec_buenas1284,ind_buenas,colores_map_rgb,error32,A,Ap32,Ap_net,Afit32] = analysis_others(file,fsx,xmax,f_lim,x_lim,b,umbral_error,size_dots_x,size_dots_f,fig,size_dots_x2);





figure(1),plot(xy_buenas064(:,1),xy_buenas064(:,2),'b.'),axis([0 0.64 0 0.64]),axis square
title('Original nonlinearity (square-root)')
figure(2),plot(xy_buenas0642(:,1),xy_buenas0642(:,2),'b.'),axis([0 0.64 0 0.64]),axis square
title('Lower exponent')
figure(3),plot(xy_buenas0643(:,1),xy_buenas0643(:,2),'b.'),axis([0 0.64 0 0.64]),axis square
title('Higher exponent')
figure(4),plot(xy_buenas0644(:,1),xy_buenas0644(:,2),'b.'),axis([0 0.64 0 0.64]),axis square
title('Arctan')



% save posiciones xy* frec*
% 
% figure(1),plot(xy_buenas064(:,1),xy_buenas064(:,2),'b.'),axis([0 0.64 0 0.64]),axis square
% figure(2),plot(xy_buenas080(:,1),xy_buenas080(:,2),'b.'),axis([0 0.80 0 0.80]),axis square
% figure(3),plot(xy_buenas128(:,1),xy_buenas128(:,2),'b.'),axis([0 1.28 0 1.28]),axis square
% figure(4),plot(xy_buenas200(:,1),xy_buenas200(:,2),'b.'),axis([0 2.00 0 2.00]),axis square
% figure(5),plot(xy_buenas256A(:,1),xy_buenas256A(:,2),'b.'),axis([0 2.56 0 2.56]),axis square
% figure(6),plot(xy_buenas400A(:,1),xy_buenas400A(:,2),'b.'),axis([0 4.00 0 4.00]),axis square
% figure(7),plot(xy_buenas512A(:,1),xy_buenas512A(:,2),'b.'),axis([0 5.12 0 5.12]),axis square
% figure(8),plot(xy_buenas800A(:,1),xy_buenas800A(:,2),'b.'),axis([0 8.00 0 8.00]),axis square
% tile

% GUARDAR FIGURAS

% MAIN RESULT

h=figure(101),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f101','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16_sin.pdf','-dpdf')
  savefig(101,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16_sin.fig')  

h=figure(107),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f107','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.pdf','-dpdf')
  savefig(107,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.fig')  
  
h=figure(113),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f113','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32_sin.pdf','-dpdf')
  savefig(113,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32_sin.fig')  

h=figure(129),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f129','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50_sin.pdf','-dpdf')
  savefig(129,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50_sin.fig')  
    
h=figure(135),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f135','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A_sin.pdf','-dpdf')
  savefig(135,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A_sin.fig')  

h=figure(141),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f141','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A_sin.pdf','-dpdf')
  savefig(141,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A_sin.fig')  

h=figure(147),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f147','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A_sin.pdf','-dpdf')
  savefig(147,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A_sin.fig')  
  
h=figure(153),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f153','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A_sin.pdf','-dpdf')
  savefig(153,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A_sin.fig')  

% 
% h=figure(101),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f101','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16_sin.pdf','-dpdf','-fillpage')
%   savefig(101,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16_sin.fig')  
% 
% h=figure(107),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f107','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.pdf','-dpdf','-fillpage')
%   savefig(107,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.fig')  
%   
% h=figure(113),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f113','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32_sin.pdf','-dpdf','-fillpage')
%   savefig(113,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32_sin.fig')  
% 
% h=figure(129),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f129','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50_sin.pdf','-dpdf','-fillpage')
%   savefig(129,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50_sin.fig')  
%     
% h=figure(135),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f135','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A_sin.pdf','-dpdf','-fillpage')
%   savefig(135,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A_sin.fig')  
% 
% h=figure(141),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f141','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A_sin.pdf','-dpdf','-fillpage')
%   savefig(141,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A_sin.fig')  
% 
% h=figure(147),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f147','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A_sin.pdf','-dpdf','-fillpage')
%   savefig(147,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A_sin.fig')  
%   
% h=figure(153),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f153','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A_sin.pdf','-dpdf','-fillpage')
%   savefig(153,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A_sin.fig')  
%   
% 
% h=figure(101),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f101','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16_sin.pdf','-dpdf','-bestfit')
%   savefig(101,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16_sin.fig')  
% 
% h=figure(107),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f107','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.pdf','-dpdf','-bestfit')
%   savefig(107,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.fig')  
%   
% h=figure(113),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f113','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32_sin.pdf','-dpdf','-bestfit')
%   savefig(113,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32_sin.fig')  
% 
% h=figure(129),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f129','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50_sin.pdf','-dpdf','-bestfit')
%   savefig(129,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50_sin.fig')  
%     
% h=figure(135),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f135','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A_sin.pdf','-dpdf','-bestfit')
%   savefig(135,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A_sin.fig')  
% 
% h=figure(141),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f141','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A_sin.pdf','-dpdf','-bestfit')
%   savefig(141,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A_sin.fig')  
% 
% h=figure(147),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f147','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A_sin.pdf','-dpdf','-bestfit')
%   savefig(147,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A_sin.fig')  
%   
% h=figure(153),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
% print('-f153','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A_sin.pdf','-dpdf','-bestfit')
%   savefig(153,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A_sin.fig')  
   
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% 16
h=figure(1),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f1','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_16.pdf','-dpdf')
  savefig(1,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_16.fig')

h=figure(6),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f6','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_16.pdf','-dpdf')
  savefig(6,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_16.fig')  
  
h=figure(3),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f3','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16.pdf','-dpdf')
  savefig(3,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_16.fig')

h=figure(4),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f4','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_16.pdf','-dpdf')
  savefig(4,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_16.fig')
  
%%%%%%%%%%%%% 20
h=figure(7),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f7','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_20.pdf','-dpdf')
  savefig(7,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_20.fig')

h=figure(12),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f12','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_20.pdf','-dpdf')
  savefig(12,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_20.fig')  
  
h=figure(9),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f9','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20.pdf','-dpdf')
  savefig(9,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20.fig')
  
h=figure(10),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f10','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_20.pdf','-dpdf')
  savefig(10,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_20.fig')
  
h=figure(107),set(h,'PaperUnits','centimeters','PaperSize',[8.4 8.4],'PaperPosition',[-1.2 -1.35 10.7 10.7])  
print('-f107','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.pdf','-dpdf')
  savefig(107,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_20_sin.fig')  
 
%%%%%%%%%%%%% 32
h=figure(13),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f13','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_32.pdf','-dpdf')
  savefig(13,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_32.fig')

h=figure(18),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f18','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_32.pdf','-dpdf')
  savefig(18,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_32.fig')  
  
h=figure(15),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f15','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32.pdf','-dpdf')
  savefig(15,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_32.fig')
  
h=figure(16),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f16','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_32.pdf','-dpdf')
  savefig(16,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_32.fig')
  
%%%%%%%%%%%%% 32 2x
h=figure(17),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f17','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_322.pdf','-dpdf')
  savefig(17,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_322.fig')

h=figure(22),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f22','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_322.pdf','-dpdf')
  savefig(22,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_322.fig')  
  
h=figure(19),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f19','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322.pdf','-dpdf')
  savefig(19,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322.fig')
  
h=figure(20),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f20','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_322.pdf','-dpdf')
  savefig(20,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_322.fig')

%%%%%%%%%%%%% 32 4x
h=figure(23),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f23','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_324.pdf','-dpdf')
  savefig(23,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_324.fig')

h=figure(28),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f28','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_324.pdf','-dpdf')
  savefig(28,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_324.fig')  
  
h=figure(25),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f25','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324.pdf','-dpdf')
  savefig(25,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324.fig')
  
h=figure(26),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f26','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_324.pdf','-dpdf')
  savefig(26,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_324.fig')
  
%%%%%%%%%%%%% 50
h=figure(29),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f29','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_50.pdf','-dpdf')
  savefig(29,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_50.fig')

h=figure(34),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f34','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_50.pdf','-dpdf')
  savefig(34,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_50.fig')  
  
h=figure(31),set(h,'PaperUnits','centimeters','PaperSize',[15 15],'PaperPosition',[-0 -0 1.5*10.7 1.5*10.7]) 
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f31','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50.pdf','-dpdf')
%print('-f31','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50.eps','-depsc2')
  savefig(31,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50.fig')
  
h=figure(32),set(h,'PaperUnits','centimeters','PaperSize',[15 15],'PaperPosition',[-0 -0 1.5*10.7 1.5*10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f32','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_50.pdf','-dpdf')
%print('-f32','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_50.eps','-depsc2')
  savefig(32,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_50.fig')
  
h=figure(31),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f31','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50b.pdf','-dpdf')
%print('-f31','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50.eps','-depsc2')
  savefig(31,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_50b.fig')
  
h=figure(32),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f32','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_50b.pdf','-dpdf')
%print('-f32','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_50.eps','-depsc2')
  savefig(32,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_50b.fig')  
  
%%%%%%%%%%%%% 32 2x AA
h=figure(35),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f35','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_322A.pdf','-dpdf')
  savefig(35,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_322A.fig')

h=figure(40),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f40','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_322A.pdf','-dpdf')
  savefig(40,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_322A.fig')  
  
h=figure(37),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f37','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A.pdf','-dpdf')
  savefig(37,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_322A.fig')
  
h=figure(38),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f38','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_322A.pdf','-dpdf')
  savefig(38,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_322A.fig')

%%%%%%%%%%%%% 32 4x AA
h=figure(41),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f41','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_324A.pdf','-dpdf')
  savefig(41,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_324A.fig')

h=figure(46),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f46','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_324A.pdf','-dpdf')
  savefig(46,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_324A.fig')  
  
h=figure(43),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f43','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A.pdf','-dpdf')
  savefig(43,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_324A.fig')
  
h=figure(44),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f44','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_324A.pdf','-dpdf')
  savefig(44,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_324A.fig')

%%%%%%%%%%%%% 50 2x  AA
h=figure(47),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f47','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_502A.pdf','-dpdf')
  savefig(47,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_502A.fig')

h=figure(52),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f52','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_502A.pdf','-dpdf')
  savefig(52,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_502A.fig')  
  
h=figure(49),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7])  
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f49','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A.pdf','-dpdf')
  savefig(49,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_502A.fig')
  
h=figure(50),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f50','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_502A.pdf','-dpdf')
  savefig(50,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_502A.fig')

%%%%%%%%%%%%% 50 4x AA
h=figure(53),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f53','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_5024A.pdf','-dpdf')
  savefig(53,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_504A.fig')

h=figure(58),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f58','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_504A.pdf','-dpdf')
  savefig(58,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_504A.fig')  
  
h=figure(55),set(h,'PaperUnits','centimeters','PaperSize',[9.5 9.7],'PaperPosition',[-0.35 -1.0 10.7 10.7])  
            set(gca,'XTick',[0 2 4 6 8],'YTick',[0 2 4 6 8])
print('-f55','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A.pdf','-dpdf')
  savefig(55,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_504A.fig')
  
h=figure(56),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
            set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f56','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_504A.pdf','-dpdf')
  savefig(56,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_504A.fig')

close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RESULTADOS 100
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = 50;
file = 'sortu_1x_100_1';
fsx = 25;
xmax = 5.12;
f_lim = fsx/2;
x_lim = 4;
b = 3;
umbral_error1 = 0.1;
size_dots_x = 9;
size_dots_f = 11;
func_criticas = [8 10;9 9;9 10;9 11;10 8;10 10;10 11;10 12;11 8;11 9;11 11;12 8;12 9;12 10;12 11;13 10;13 11;13 12;9 20;9 21;7 25;9 25;10 25;11 24];
[xf_marina1,colores_map_rgb_buenos1,error_bueno1,error1,Afit_manual1,Ap1]=analysis_100_manual(file,fsx,umbral_error1,func_criticas,fig);

file = 'sortu_1x_100_2';
fsx = 25;
xmax = 5.12;
f_lim = fsx/2;
x_lim = 4;
b = 3;
umbral_error1 = 0.1;
size_dots_x = 9;
size_dots_f = 11;
func_criticas = [5 5];
fig = 5;
[xf_marina2,colores_map_rgb_buenos2,error_bueno2,error2,Afit_manual2,Ap2]=analysis_100_manual(file,fsx,umbral_error1,func_criticas,fig);

file = 'sortu_1x_100_3';
fsx = 25;
xmax = 5.12;
f_lim = fsx/2;
x_lim = 4;
b = 3;
umbral_error1 = 0.1;
size_dots_x = 9;
size_dots_f = 11;
func_criticas = [8 6;8 7;8 8;9 8;9 9;11 8];
fig = 9;
[xf_marina3,colores_map_rgb_buenos3,error_bueno3,error3,Afit_manual3,Ap3]=analysis_100_manual(file,fsx,umbral_error1,func_criticas,fig);

file = 'sortu_1x_100_4';
fsx = 25;
xmax = 4;
f_lim = fsx/2;
x_lim = 4;
b = 3;
umbral_error1 = 0.1;
size_dots_x = 9;
size_dots_f = 11;
func_criticas = [7 10;9 7;9 8;9 9;10 8];
fig = 13;
[xf_marina4,colores_map_rgb_buenos4,error_bueno4,error4,Afit_manual4,Ap4]=analysis_100_manual(file,fsx,umbral_error1,func_criticas,fig);

xf_mar = [xf_marina1;xf_marina2;xf_marina3;xf_marina4];
colores = [colores_map_rgb_buenos1 colores_map_rgb_buenos2 colores_map_rgb_buenos3 colores_map_rgb_buenos4];
errores = [error1 error2 error3 error4];

ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);

fig=100;
figure(fig+2)
xf_sort = xf_mar;
colores_map_rgb_buenos = colores;
for i=1:size(xf_sort,1)
    x=xf_sort(i,1);
    y=xf_sort(i,2);
    z=colores_map_rgb_buenos(i);
       if (x < x_lim) & (y < x_lim)
          plot(x,y,'.','markersize',(0.64/4)*140/1.5,'color',map_rgb(z,:)), hold on
       end
end
axis([-0.01 4.01 -0.01 4.01]),axis square,axis ij
xlabel('x (deg)'),ylabel('y (deg)'),
set(gca,'Xaxislocation','top')
set(gca,'XTick',[0 1 2 3 4],'YTick',[0 1 2 3 4])
%title('Centers of TICA sensors in the retinal space')
set(gcf,'color',[1 1 1])

f_lim = 9;
figure(fig+3)
for i=1:size(xf_sort,1)
    z=colores_map_rgb_buenos(i);
        plot(xf_sort(i,3),xf_sort(i,4),'.','markersize',size_dots_f,'color',map_rgb(z,:)), hold on 
        plot(-xf_sort(i,3),-xf_sort(i,4),'.','markersize',size_dots_f,'color',map_rgb(z,:)), hold on 
end
axis([-f_lim f_lim -f_lim f_lim])
axis square
%title('Centers of TICA features in the frequency domain')
xlabel('f_x (c/deg)'),ylabel('f_y (c/deg)'),
set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
set(gcf,'color',[1 1 1])

figure(fig+4),colormap gray,imagesc(Ap1(103:2165,1:3093)),axis off,axis equal
figure(fig+5),colormap gray,imagesc(Afit_manual1(103:2165,1:3093)),axis off,axis equal

figure(fig+6),colormap gray,imagesc(Ap2),axis off,axis equal
figure(fig+7),colormap gray,imagesc(Afit_manual2),axis off,axis equal

figure(fig+8),colormap gray,imagesc(Ap3),axis off,axis equal
figure(fig+9),colormap gray,imagesc(Afit_manual3),axis off,axis equal

figure(fig+10),colormap gray,imagesc(Ap4),axis off,axis equal
figure(fig+11),colormap gray,imagesc(Afit_manual4),axis off,axis equal


%%%%%%%%%%%%% 100
h=figure(102),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -1.2 10.7 10.7])  
set(gca,'XTick',[0 1 2 3 4],'YTick',[0 1 2 3 4])
print('-f102','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100.pdf','-dpdf')
  savefig(102,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100.fig')
  
h=figure(103),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 10.7 10.7]) 
set(gca,'XTick',[-8 -6 -4 -2 0 2 4 6 8],'YTick',[-8 -6 -4 -2 0 2 4 6 8])
print('-f103','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100.pdf','-dpdf')
  savefig(103,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100.fig')
  
%%%%%%%%%%%%% 100-1
h=figure(104),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f104','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_1.pdf','-dpdf')
  savefig(104,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_1.fig')

h=figure(105),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f105','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_1.pdf','-dpdf')
  savefig(105,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_1.fig')  
  
h=figure(52),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f52','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_1.pdf','-dpdf')
  savefig(52,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_1.fig')
  
h=figure(53),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[-12 -9 -6 -3 0 3 6 9 12],'YTick',[-12 -9 -6 -3 0 3 6 9 12])
print('-f53','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_1.pdf','-dpdf')
  savefig(53,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_1.fig')  
  
%%%%%%%%%%%%% 100-2
h=figure(106),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f106','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_2.pdf','-dpdf')
  savefig(106,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_2.fig')

h=figure(107),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f107','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_2.pdf','-dpdf')
  savefig(107,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_2.fig')  
  
h=figure(7),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f7','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_2.pdf','-dpdf')
  savefig(7,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_2.fig')
  
h=figure(8),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[-12 -9 -6 -3 0 3 6 9 12],'YTick',[-12 -9 -6 -3 0 3 6 9 12])
print('-f8','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_2.pdf','-dpdf')
  savefig(8,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_2.fig')

%%%%%%%%%%%%% 100-3
h=figure(108),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f108','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_3.pdf','-dpdf')
  savefig(108,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_3.fig')

h=figure(109),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f109','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_3.pdf','-dpdf')
  savefig(109,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_3.fig')  
  
h=figure(11),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f11','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_3.pdf','-dpdf')
  savefig(11,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_3.fig')
  
h=figure(12),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[-12 -9 -6 -3 0 3 6 9 12],'YTick',[-12 -9 -6 -3 0 3 6 9 12])
print('-f12','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_3.pdf','-dpdf')
  savefig(12,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_3.fig')
  
%%%%%%%%%%%%% 100-4
h=figure(110),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])
print('-f110','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_4.pdf','-dpdf')
  savefig(110,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_100_4.fig')

h=figure(111),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-1.2 -1.2 12 12])  
  print('-f111','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_4.pdf','-dpdf')
  savefig(111,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\features_fit_100_4.fig')  
  
h=figure(15),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[0 1 2 3 4 5],'YTick',[0 1 2 3 4 5])
print('-f15','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_4.pdf','-dpdf')
  savefig(15,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_espacial_100_4.fig')
  
h=figure(16),set(h,'PaperUnits','centimeters','PaperSize',[10 10],'PaperPosition',[-0 -0 1*10.7 1*10.7]) 
            set(gca,'XTick',[-12 -9 -6 -3 0 3 6 9 12],'YTick',[-12 -9 -6 -3 0 3 6 9 12])
print('-f16','C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_4.pdf','-dpdf')
  savefig(16,'C:\disco_portable\mundo_irreal\latex\VisRes_16\figures2\dist_fourier_100_4.fig')
  
% % ERRORES (con aliasing)
%   
% h=figure(fig+12),plot(sort(error16),'linewidth',2),xlabel('TICA feature'),ylabel('Mean Absolute Error / Range'),set(h,'color',[1 1 1])
% hold on,plot(sort(error20),'linewidth',2),
% plot(sort(error32),'linewidth',2),
% plot(sort(error50),'linewidth',2),
% plot(sort(error322),'linewidth',2),
% plot(sort(errores),'-.','linewidth',2),
% plot(sort(error324),'linewidth',2),
% plot(1:length(errores),0.125*ones(length(errores)),'k-')
% legend('\Delta x = 0.64 deg, f_s = 25 cpd, 16 x 16 pixels','\Delta x = 0.80 deg, f_s = 25 cpd, 20 x 20 pixels','\Delta x = 1.28 deg, f_s = 25 cpd, 32 x 32 pixels','\Delta x = 2.00 deg, f_s = 25 cpd, 50 x 50 pixels','\Delta x = 2.56 deg, f_s = 12 cpd, 32 x 32 pixels','\Delta x = 4.00 deg, f_s = 25 cpd, 100 x 100 pixels','\Delta x = 5.12 deg, f_s = 6  cpd, 32 x 32 pixels','Error Threshold')
% legend boxoff
% axis([0 length(errores) 0 0.45])
% % title('Gabor Fitting Errors')

% ERRORES (sin aliasing)
  
h=figure(fig+13),plot(sort(error16),'linewidth',2,'color',[1 0 0]),xlabel('TICA feature'),ylabel('Mean Absolute Error / Range'),set(h,'color',[1 1 1])
hold on,h=plot(sort(error20),'linewidth',2,'color',[0.8 0 0]),
plot(sort(error32),'linewidth',2,'color',[0.6 0 0]),
plot(sort(error50),'linewidth',2,'color',[0.4 0 0]),
plot(sort(error322_A),'linewidth',2,'color',[0 0.9 0]),
plot(sort(errores),'-.','linewidth',2,'color',[0 0 1]),
plot(sort(error502_A),'linewidth',2,'color',[0 0.75 0]),
plot(sort(error324_A),'linewidth',2,'color',[0 0.55 0]),
plot(sort(error504_A),'linewidth',2,'color',[0 0.35 0]),
plot(1:length(errores),0.125*ones(length(errores)),'k-')
legend('0.64 deg, f_s = 25 cpd, 16 x 16 pixels','0.80 deg, f_s = 25 cpd, 20 x 20 pixels','1.28 deg, f_s = 25 cpd, 32 x 32 pixels','2.00 deg, f_s = 25 cpd, 50 x 50 pixels',...
       '2.56 deg, f_s = 12 cpd, 32 x 32 pixels','4.00 deg, f_s = 25 cpd, 100 x 100 pixels','4.00 deg, f_s = 12 cpd, 50 x 50 pixels','5.12 deg, f_s = 6  cpd, 32 x 32 pixels',...
       '8.00 deg, f_s = 6  cpd, 50 x 50 pixels','Error Threshold')
legend boxoff
axis([0 length(errores) 0 0.45])

%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% CONVERGENCIA
%%%%%%%%%%%%%%%%%%%%


load tica_1x_16_image objiter obj;objiter16=objiter;obj16=obj;
load tica_1x_20_image objiter obj;objiter20=objiter;obj20=obj;
load tica_1x_32_image objiter obj;objiter32=objiter;obj32=obj;
load tica_1x_50_image objiter obj;objiter50=objiter;obj50=obj;
load tica_2x_32_image_A objiter obj;objiter322=objiter;obj322=obj;
load tica_2x_50_image_A objiter obj;objiter502=objiter;obj502=obj;
%load tica_1x_100_image objiter obj;objiter100=objiter;obj100=obj;
load iteraciones100 objiter obj;objiter100=objiter;obj100=obj;
load tica_4x_32_image_A objiter obj;objiter324=objiter;obj324=obj;
load tica_4x_50_image_A objiter obj;objiter504=objiter;obj504=obj;

h=figure(300),semilogx(objiter16,obj16/obj16(1),'linewidth',2,'color',[1 0 0]),xlabel('Iteration'),ylabel('Normalized TICA Goal Function'),set(h,'color',[1 1 1])
hold on,h=semilogx(objiter20,obj20/obj20(1),'linewidth',2,'color',[0.8 0 0]),
semilogx(objiter32,obj32/obj32(1),'linewidth',2,'color',[0.6 0 0]),
semilogx(objiter50,obj50/obj50(1),'linewidth',2,'color',[0.4 0 0]),
semilogx(objiter322,obj322/obj322(1),'linewidth',2,'color',[0 0.9 0]),
semilogx(objiter100,obj100/obj100(1),'-.','linewidth',2,'color',[0 0 1]),
semilogx(objiter502,obj502/obj502(1),'linewidth',2,'color',[0 0.75 0]),
semilogx(objiter324,obj324/obj324(1),'linewidth',2,'color',[0 0.55 0]),
semilogx(objiter504,obj504/obj504(1),'linewidth',2,'color',[0 0.35 0]),
% plot(1:length(errores),0.125*ones(length(errores)),'k-')
legend('0.64 deg, f_s = 25 cpd, 16 x 16 pixels','0.80 deg, f_s = 25 cpd, 20 x 20 pixels','1.28 deg, f_s = 25 cpd, 32 x 32 pixels','2.00 deg, f_s = 25 cpd, 50 x 50 pixels',...
       '2.56 deg, f_s = 12 cpd, 32 x 32 pixels','4.00 deg, f_s = 25 cpd, 100 x 100 pixels','4.00 deg, f_s = 12 cpd, 50 x 50 pixels','5.12 deg, f_s = 6  cpd, 32 x 32 pixels',...
       '8.00 deg, f_s = 6  cpd, 50 x 50 pixels')
legend boxoff
axis([1 2200 0.925 1.005])

%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% DATABASE
%%%%%%%%%%%%%%%%%%%%%

% uiopen('C:\disco_portable\mundo_irreal\latex\VisRes_16\code\ICA\imageica_00\data\5.tiff',1)

x5 = imread('C:\disco_portable\mundo_irreal\latex\VisRes_16\code\ICA\imageica_00\data\5.tiff');

% %centro = [152 394]
% centro = [157 394]
% %centro = [160 370]
% s=8;
% figure(5),colormap gray,imagesc(linspace(0,0.64,2*s),linspace(0,0.64,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square
% s=10;
% figure(6),colormap gray,imagesc(linspace(0,0.8,2*s),linspace(0,0.8,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square
% s=16;
% figure(7),colormap gray,imagesc(linspace(0,1.28,2*s),linspace(0,1.28,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square
% s=25;
% figure(8),colormap gray,imagesc(linspace(0,2,2*s),linspace(0,2,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square
% s=50;
% figure(9),colormap gray,imagesc(linspace(0,4,2*s),linspace(0,4,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square
% 
% x52x = x5(1:2:end,1:2:end);
% centro2 = centro/2;
% s=16;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
% 
% x54x = x5(1:4:end,1:4:end);
% %x54x = imresize(x5,0.25,'bilinear')
% centro4 = centro/4;
% s=16;
% figure(2),colormap gray,imagesc(linspace(0,5.12,2*s),linspace(0,5.12,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square
% 
% %%%%%%%%%%%%%%%%%%%%%%%
% 
% centro = [157 394]
% %centro = [160 370]
% s=8;
% figure(100),subplot(1,7,1),colormap gray,imagesc(linspace(0,0.64,2*s),linspace(0,0.64,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 0.64 deg';'f_s = 25 cpd';'16 x 16 pix'})
% s=10;
% figure(100),subplot(1,7,2),colormap gray,imagesc(linspace(0,0.8,2*s),linspace(0,0.8,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 0.80 deg';'f_s = 25 cpd';'20 x 20 pix'})
% s=16;
% figure(100),subplot(1,7,3),colormap gray,imagesc(linspace(0,1.28,2*s),linspace(0,1.28,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 1.28 deg';'f_s = 25 cpd';'32 x 32 pix'})
% s=25;
% figure(100),subplot(1,7,4),colormap gray,imagesc(linspace(0,2,2*s),linspace(0,2,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 2.00 deg';'f_s = 25 cpd';'50 x 50 pix'})
% 
% x52x = x5(1:2:end,1:2:end);
% centro2 = centro/2;
% s=16;
% figure(100),subplot(1,7,5),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 2.56 deg';'f_s = 12.5 cpd';'32 x 32 pix'})
% 
% s=50;
% figure(100),subplot(1,7,6),colormap gray,imagesc(linspace(0,4,2*s),linspace(0,4,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 4.00 deg';'f_s = 25 cpd';'100x100 pix'})
% 
% x54x = x5(1:4:end,1:4:end);
% %x54x = imresize(x5,0.25,'bilinear')
% centro4 = centro/4;
% s=16;
% figure(100),subplot(1,7,7),colormap gray,imagesc(linspace(0,5.12,2*s),linspace(0,5.12,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,axis off,title({'\Deltax = 5.12 deg';'f_s = 6.25 cpd';'32 x 32 pix'})

% %%%%%%%%%%%%%%%%%%%%% DATABASE 1 CON ALIASING
% 
% tamanyos = get(0,'ScreenSize'); % left, bottom, width, height
% ancho = round(0.85*tamanyos(3));
% alto = round(0.85*tamanyos(4));
% ratio = ancho/alto;
% 
% h0=figure(100);
% pos_fig = [round(0.05*tamanyos(3)) round(0.05*tamanyos(4)) ancho alto];
% set(h0,'Position',pos_fig,'color',[1 1 1])
% 
% centro = [157 394]
% %centro = [160 370]
% s=8;
% tam_deg = 0.64;
% alto_im = 0.03*(tam_deg/0.64);
% h1=subplot(1,7,1);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 0.64 deg';'f_s = 25 cpd';'16x16 pix'})
% set(h1,'Position',[0.025 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=10;
% tam_deg = 0.8;
% alto_im = 0.03*(tam_deg/0.64);
% h2=subplot(1,7,2);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 0.80 deg';'f_s = 25 cpd';'20x20 pix'})
% set(h2,'Position',[0.09 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=16;
% tam_deg = 1.28;
% alto_im = 0.03*(tam_deg/0.64);
% h3=subplot(1,7,3);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 1.28 deg';'f_s = 25 cpd';'32x32 pix'})
% set(h3,'Position',[0.16 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=25;
% tam_deg = 2;
% alto_im = 0.03*(tam_deg/0.64);
% h4=subplot(1,7,4);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 2.00 deg';'f_s = 25 cpd';'50x50 pix'})
% set(h4,'Position',[0.25 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% x52x = x5(1:2:end,1:2:end);
% centro2 = centro/2;
% s=16;
% % figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
% tam_deg = 2.56;
% alto_im = 0.03*(tam_deg/0.64);
% h5 = subplot(1,7,5);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 2.56 deg';'f_s = 12.5 cpd';'32x32 pix'})
% set(h5,'Position',[0.375 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=50;
% tam_deg = 4;
% alto_im = 0.03*(tam_deg/0.64);
% h6=subplot(1,7,6);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 4.00 deg';'f_s = 25 cpd';'100x100 pix'})
% set(h6,'Position',[0.525 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% x54x = x5(1:4:end,1:4:end);
% centro4 = centro/4;
% s=16;
% % figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
% tam_deg = 5.12;
% alto_im = 0.03*(tam_deg/0.64);
% h7 = subplot(1,7,7);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 5.12 deg';'f_s = 6.25 cpd';'32x32 pix'})
% set(h7,'Position',[0.75 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);

% %%%%%%%%%%%%%%%%%%%%%  DATABASE 1 SIN ALIASING
% 
% tamanyos = get(0,'ScreenSize'); % left, bottom, width, height
% ancho = round(0.85*tamanyos(3));
% alto = round(0.85*tamanyos(4));
% ratio = ancho/alto;
% 
% h0=figure(101);
% pos_fig = [round(0.05*tamanyos(3)) round(0.05*tamanyos(4)) ancho alto];
% set(h0,'Position',pos_fig,'color',[1 1 1])
% 
% centro = [157 394]
% %centro = [160 370]
% s=8;
% tam_deg = 0.64;
% alto_im = 0.03*(tam_deg/0.64);
% h1=subplot(1,7,1);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 0.64 deg';'f_s = 25 cpd';'16x16 pix'})
% set(h1,'Position',[0.025 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=10;
% tam_deg = 0.8;
% alto_im = 0.03*(tam_deg/0.64);
% h2=subplot(1,7,2);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 0.80 deg';'f_s = 25 cpd';'20x20 pix'})
% set(h2,'Position',[0.09 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=16;
% tam_deg = 1.28;
% alto_im = 0.03*(tam_deg/0.64);
% h3=subplot(1,7,3);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 1.28 deg';'f_s = 25 cpd';'32x32 pix'})
% set(h3,'Position',[0.16 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=25;
% tam_deg = 2;
% alto_im = 0.03*(tam_deg/0.64);
% h4=subplot(1,7,4);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 2.00 deg';'f_s = 25 cpd';'50x50 pix'})
% set(h4,'Position',[0.25 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% %x52x = x5(1:2:end,1:2:end);
% x52x = imresize(x5,0.5,'bilinear');
% centro2 = centro/2;
% s=16;
% % figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
% tam_deg = 2.56;
% alto_im = 0.03*(tam_deg/0.64);
% h5 = subplot(1,7,5);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 2.56 deg';'f_s = 12.5 cpd';'32x32 pix'})
% set(h5,'Position',[0.375 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% s=50;
% tam_deg = 4;
% alto_im = 0.03*(tam_deg/0.64);
% h6=subplot(1,7,6);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 4.00 deg';'f_s = 25 cpd';'100x100 pix'})
% set(h6,'Position',[0.525 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);
% 
% %x54x = x5(1:4:end,1:4:end);
% x54x = imresize(x5,0.25,'bilinear');
% centro4 = centro/4;
% s=16;
% % figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
% tam_deg = 5.12;
% alto_im = 0.03*(tam_deg/0.64);
% h7 = subplot(1,7,7);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,axis off
% title({'\Deltax = 5.12 deg';'f_s = 6.25 cpd';'32x32 pix'})
% set(h7,'Position',[0.75 0.5-ratio*(alto_im/2) alto_im ratio*alto_im]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATABASE 2 con la de 100
% 

tamanyos = get(0,'ScreenSize'); % left, bottom, width, height
ancho = round(0.95*tamanyos(3));
alto = round(0.95*tamanyos(4));
ratio = ancho/alto;

h0=figure(200);
pos_fig = [round(0.05*tamanyos(3)) round(0.05*tamanyos(4)) ancho alto];
set(h0,'Position',pos_fig,'color',[1 1 1])

factor = 0.0255;
desp1 = 0.022;
desp2 = 0.045;
desp3 = 0.075;
desp4 = 0.11;
posiciones = [0.025 0.09-desp1 0.16-desp2 0.25-desp3 0.375-desp4 0.525-desp4 0.75-desp4];

desp = 0.01;
posiciones = [0.025 0.068 0.92*[0.115 0.175 0.265 0.375] 0.97*0.52 0.99*0.71]-desp;
altura = 0.65;

centro = [157 394]
%centro = [160 370]
s=8;
tam_deg = 0.64;
alto_im = factor*(tam_deg/0.64);
h1=subplot(1,8,1);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
title('0.64 deg')
xlabel({'f_s=25 cpd';'16x16 pix'})
set(h1,'Position',[posiciones(1) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=10;
tam_deg = 0.8;
alto_im = factor*(tam_deg/0.64);
h2=subplot(1,8,2);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%%axis off
title('0.80 deg')
xlabel({'f_s=25 cpd';'20x20 pix'})
set(h2,'Position',[posiciones(2) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=16;
tam_deg = 1.28;
alto_im = factor*(tam_deg/0.64);
h3=subplot(1,8,3);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 1.28 deg';'f_s=25 cpd';'32x32 pix'})
t1=title('1.28 deg')
t2=xlabel({'f_s=25 cpd';'32x32 pix'})
set(t1,'color',[1 0 0]);set(t2,'color',[1 0 0])
set(h3,'Position',[posiciones(3) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=25;
tam_deg = 2;
alto_im = factor*(tam_deg/0.64);
h4=subplot(1,8,4);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 2.00 deg';'f_s=25 cpd';'50x50 pix'})
title('2.00 deg')
xlabel({'f_s=25 cpd';'50x50 pix'})
set(h4,'Position',[posiciones(4) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

%x52x = x5(1:2:end,1:2:end);
x52x = imresize(x5,0.5,'bilinear');
centro2 = centro/2;
s = 16;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
tam_deg = 2.56;
alto_im = factor*(tam_deg/0.64);
h5 = subplot(1,8,5);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 2.56 deg';'f_s=12.5 cpd';'32x32 pix'})
title('2.56 deg')
xlabel({'f_s=12.5 cpd';'32x32 pix'})
set(h5,'Position',[posiciones(5) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

x52x = imresize(x5,0.5,'bilinear');
centro2 = centro/2;
s = 25;
tam_deg = 4;
alto_im = factor*(tam_deg/0.64);
h4=subplot(1,8,6);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 4.00 deg';'f_s=12.5 cpd';'50x50 pix'})
t1=title('4.00 deg')
t2=xlabel({'f_s=12.5 cpd';'50x50 pix'})
set(t1,'color',[0 0 1]);set(t2,'color',[0 0 1])
set(h4,'Position',[posiciones(6) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=50;
tam_deg = 4;
alto_im = factor*(tam_deg/0.64);
h6=subplot(1,8,6);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 4.00 deg';'f_s=25 cpd';'100x100 pix'})
%title('4.00 deg')
t1=xlabel({'f_s=25 cpd';'100x100 pix'})
set(t1,'color',[0 0 1]);
set(h6,'Position',[posiciones(6) (altura-0.34)-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

%x54x = x5(1:4:end,1:4:end);
x54x = imresize(x5,0.25,'bilinear');
centro4 = centro/4;
s=16;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
tam_deg = 5.12;
alto_im = factor*(tam_deg/0.64);
h7 = subplot(1,8,7);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 5.12 deg';'f_s=6.25 cpd';'32x32 pix'})
title('5.12 deg')
xlabel({'f_s=6.25 cpd';'32x32 pix'})
set(h7,'Position',[posiciones(7) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

%x54x = x5(1:4:end,1:4:end);
x54x = imresize(x5,0.25,'bilinear');
centro4 = centro/4;
s=25;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
tam_deg = 8;
alto_im = factor*(tam_deg/0.64);
h7 = subplot(1,8,8);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 5.12 deg';'f_s=6.25 cpd';'32x32 pix'})
title('8.00 deg')
xlabel({'f_s=6.25 cpd';'50x50 pix'})
set(h7,'Position',[posiciones(8) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATABASE 2 SIN LA DE 100
% 

tamanyos = get(0,'ScreenSize'); % left, bottom, width, height
ancho = round(0.95*tamanyos(3));
alto = round(0.95*tamanyos(4));
ratio = ancho/alto;

h0=figure(200);
pos_fig = [round(0.05*tamanyos(3)) round(0.05*tamanyos(4)) ancho alto];
set(h0,'Position',pos_fig,'color',[1 1 1])

factor = 0.0255;
desp1 = 0.022;
desp2 = 0.045;
desp3 = 0.075;
desp4 = 0.11;
posiciones = [0.025 0.09-desp1 0.16-desp2 0.25-desp3 0.375-desp4 0.525-desp4 0.75-desp4];
ALTURA = 0.1;

desp = 0.01;
posiciones = [0.025 0.068 0.92*[0.115 0.175 0.265 0.375] 0.97*0.52 0.99*0.71]-desp;
altura = 0.65;

centro = [157 394]
%centro = [160 370]
s=8;
tam_deg = 0.64;
alto_im = factor*(tam_deg/0.64);
h1=subplot(1,8,1);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
title('0.64 deg')
xlabel({'f_s=25 cpd';'16x16 pix'})
set(h1,'Position',[posiciones(1) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h1,'Position',[posiciones(1) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=10;
tam_deg = 0.8;
alto_im = factor*(tam_deg/0.64);
h2=subplot(1,8,2);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%%axis off
title('0.80 deg')
xlabel({'f_s=25 cpd';'20x20 pix'})
set(h2,'Position',[posiciones(2) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h2,'Position',[posiciones(2) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=16;
tam_deg = 1.28;
alto_im = factor*(tam_deg/0.64);
h3=subplot(1,8,3);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 1.28 deg';'f_s=25 cpd';'32x32 pix'})
t1=title('1.28 deg')
t2=xlabel({'f_s=25 cpd';'32x32 pix'})
set(t1,'color',[1 0 0]);set(t2,'color',[1 0 0])
set(h3,'Position',[posiciones(3) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h3,'Position',[posiciones(3) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

s=25;
tam_deg = 2;
alto_im = factor*(tam_deg/0.64);
h4=subplot(1,8,4);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 2.00 deg';'f_s=25 cpd';'50x50 pix'})
title('2.00 deg')
xlabel({'f_s=25 cpd';'50x50 pix'})
set(h4,'Position',[posiciones(4) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h4,'Position',[posiciones(4) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

%x52x = x5(1:2:end,1:2:end);
x52x = imresize(x5,0.5,'bilinear');
centro2 = centro/2;
s = 16;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
tam_deg = 2.56;
alto_im = factor*(tam_deg/0.64);
h5 = subplot(1,8,5);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 2.56 deg';'f_s=12.5 cpd';'32x32 pix'})
title('2.56 deg')
xlabel({'f_s=12.5 cpd';'32x32 pix'})
set(h5,'Position',[posiciones(5) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h5,'Position',[posiciones(5) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

x52x = imresize(x5,0.5,'bilinear');
centro2 = centro/2;
s = 25;
tam_deg = 4;
alto_im = factor*(tam_deg/0.64);
h4=subplot(1,8,6);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 4.00 deg';'f_s=12.5 cpd';'50x50 pix'})
title('4.00 deg')
xlabel({'f_s=12.5 cpd';'50x50 pix'})
set(h4,'Position',[posiciones(6) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h4,'Position',[posiciones(6) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

% s=50;
% tam_deg = 4;
% alto_im = factor*(tam_deg/0.64);
% h6=subplot(1,8,6);
% colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x5(centro(1)-s+1:centro(1)+s,centro(2)-s+1:centro(2)+s),[0 255]),axis square,%axis off
% %title({'\Deltax = 4.00 deg';'f_s=25 cpd';'100x100 pix'})
% %title('4.00 deg')
% xlabel({'f_s=25 cpd';'100x100 pix'})
% set(h6,'Position',[posiciones(6) (altura-0.32)-ratio*(alto_im/2) alto_im ratio*alto_im]);
%% set(h6,'Position',[posiciones(1) ALTURA alto_im ratio*alto_im]);
% set(gca,'box','on','XTick',[],'YTick',[])

%x54x = x5(1:4:end,1:4:end);
x54x = imresize(x5,0.25,'bilinear');
centro4 = centro/4;
s=16;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
tam_deg = 5.12;
alto_im = factor*(tam_deg/0.64);
h7 = subplot(1,8,7);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 5.12 deg';'f_s=6.25 cpd';'32x32 pix'})
title('5.12 deg')
xlabel({'f_s=6.25 cpd';'32x32 pix'})
set(h7,'Position',[posiciones(7) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h7,'Position',[posiciones(7) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])

%x54x = x5(1:4:end,1:4:end);
x54x = imresize(x5,0.25,'bilinear');
centro4 = centro/4;
s=25;
% figure(10),colormap gray,imagesc(linspace(0,2.56,2*s),linspace(0,2.56,2*s),x52x(centro2(1)-s+1:centro2(1)+s,centro2(2)-s+1:centro2(2)+s),[0 255]),axis square
tam_deg = 8;
alto_im = factor*(tam_deg/0.64);
h7 = subplot(1,8,8);
colormap gray,imagesc(linspace(0,tam_deg,2*s),linspace(0,tam_deg,2*s),x54x(centro4(1)-s+1:centro4(1)+s,centro4(2)-s+1:centro4(2)+s),[0 255]),axis square,%axis off
%title({'\Deltax = 5.12 deg';'f_s=6.25 cpd';'32x32 pix'})
title('8.00 deg')
xlabel({'f_s=6.25 cpd';'50x50 pix'})
set(h7,'Position',[posiciones(8) altura-ratio*(alto_im/2) alto_im ratio*alto_im]);
set(h7,'Position',[posiciones(8) ALTURA alto_im ratio*alto_im]);
set(gca,'box','on','XTick',[],'YTick',[])