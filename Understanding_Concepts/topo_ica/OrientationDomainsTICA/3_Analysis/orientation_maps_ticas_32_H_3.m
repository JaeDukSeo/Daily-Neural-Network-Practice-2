close all%
% NON-EMERGENCE OF ORIENTATION MAPS FROM TOPOGRAPHIC ICA (IN THE RETINAL SPACE)
% 
% In this script we check how the orientation maps of the TICA topology project
% back into the retinal space. 
%
% Orientation maps certainly emerge in the TICA topology based on
% correlations between response energies [Hyvarinen&Hoyer Vis.Res.2001].
% Given the retinotopy observed in V1, the TICA orientation maps could be 
% interpreted as actual cortical orientation maps only if the retina-topology 
% transform is smooth enough (see [Bosking et al. Nature 2002] for an illustration 
% of the spatial smoothness of the retina-cortex transform).
% Figure 5b in Hyvarinen Hoyer 01 shows a diagram with the (intrincated) spatial 
% meaning of the cells in the topology, but the implications of this result were not
% discussed. 
%
% Here we analize the retinotopy of the topology in more detail by
% analyzing the distribution of the oriented sensors in the retinal space.
% This analysis gives a more clear result than the location diagram shown
% in Hyvarinen Hoyer 01. 
% Since the smooth orientation domains, in the topology, are completely scrambled 
% to give random salt-and-pepper organization, in the retina, (even the most 
% irregular reported case [Das Nature 97] is not as random) this means that 
% the topology completely violates retinotopy and hence it cannot be interpreted 
% as locations in the cortex.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% TOPOGRAPHIC ICA FROM RETINAL IMAGES 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all
%addpath(genpath('C:\disco_portable\mundo_irreal\latex\VisRes_16\code\'))
%path_result = 'C:\disco_portable\mundo_irreal\latex\VisRes_16\code\';
addpath(genpath('/media/disk/vista/Papers/PLOS_2016/code/'))
path_ima = '/media/disk/vista/Papers/PLOS_2016/code/vector_images/';
path_result = '/media/disk/vista/Papers/PLOS_2016/code/';
lados = [16 20 32 50 100];
i = 3;
factor =1;
lado = lados(i)

x = [];
for i = 1:13
    if factor == 1
       load([path_ima,'data_',num2str(lado),'_im_',num2str(i)])
    else
       load([path_ima,'data_',num2str(factor),'_',num2str(lado),'_im_',num2str(i), '_A'])  
    end
    x = [x xx];
end
N_ima=50000;
x = x(:,1:N_ima);

Ncomp = round(0.8*lado)^2;
[xw, WM, iWM, m] = whiten_hyva(  x , Ncomp );
clear x

global X   % Mierda del estimate
X = xw;

  p.seed = 1;
  p.write = 5;
  p.model = 'tica';
  p.algorithm = 'gradient';
  p.xdim = round(0.8*lado);
  % p.ydim = 10;
  p.ydim = round(0.8*lado);
  p.maptype = 'torus';
  p.neighborhood = 'ones3by3';
  p.stepsize = 0.1;
  p.epsi = 0.005;
  p.neighborhoodN = 3;
  % estimate( WM, iWM, '/media/raid5/vista/Papers/ICA_Alicante/ICA/imageica_00/results/tica_2.mat', p );
  %estimate_1( WM, iWM, [path_result,['tica_',num2str(factor),'x_',num2str(lado),'_image_nolin_1.mat']], p );
  %estimate_2( WM, iWM, [path_result,['tica_',num2str(factor),'x_',num2str(lado),'_image_nolin_2.mat']], p );
  %estimate_3( WM, iWM, [path_result,['tica_',num2str(factor),'x_',num2str(lado),'_image_nolin_3.mat']], p );
  %estimate_4( WM, iWM, [path_result,['tica_',num2str(factor),'x_',num2str(lado),'_image_nolin_4.mat']], p );

  estimate_5( WM, iWM, [path_result,['tica_',num2str(factor),'x_',num2str(lado),'_image_H_3.mat']], p );
  
%   %load
%   %/media/disk/vista/Papers/ICA_Alicante/ICA/imageica_00/results/tica_2_image.mat
%   %(se cortó ;-)
%   %
%   %load /media/disk/vista/Papers/ICA_Alicante/ICA/imageica_00/results/tica_3_image.mat 
%   % (este hizo 735 iteraciones)
% 
% figure,disp_patches(W(1:1600,:)',40,2);colormap gray,title('TICA (V1) Filters from retinal images')
% 
% fs = 50;
% [Mss,xf1_t,delt_xf_ang_phase1_t,xf2_t,delt_xf_ang_phase2_t,xfm_t,delt_xf_ang_phasem_t,er1_t,er2_t,erm_t] = sort_basis(A,fs);
% 
% ang = 0:2:359;
% map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
% map_rgb = hsv2rgb(map_hsv);
% 
% xf_sort = xfm_t;
% figure
% for i=1:1600
%     z(i)=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
% %     z(i)
% %     map_rgb(z(i),:)
%     plot(xf_sort(i,3),xf_sort(i,4),'.','markersize',20,'color',map_rgb(z(i),:)), hold on 
%     plot(-xf_sort(i,3),-xf_sort(i,4),'.','markersize',20,'color',map_rgb(z(i),:)), hold on 
%     axis([-30 30 -30 30])
%     pause(0.1)
% end
% axis square
% title('V1 features in the frequency domain (TICA)')
% xlabel('f_x (c/deg)'),ylabel('f_y (c/deg)'),
% set(gcf,'color',[1 1 1])
% 
% [Mss,xf1_t_fil,delt_xf_ang_phase1_t_fil,xf2_t_fil,delt_xf_ang_phase2_t_fil,xfm_t_fil,delt_xf_ang_phasem_t_fil,er1_t_fil,er2_t_fil,erm_t_fil] = sort_basis(W',fs);
% 
% xf_sort=xf2_t_fil;
% figure
% for i=1:1600
%     x=xf_sort(i,1);
%     y=xf_sort(i,2);
%     z=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1
%     plot(x,y,'.','markersize',20,'color',map_rgb(z,:)), hold on 
% end
% axis([-0.05 1.05 -0.05 1.05]),axis square,axis ij
% xlabel('x (deg)'),ylabel('y (deg)'),
% title('V1 sampling of the spatial domain (TICA)')
% set(gcf,'color',[1 1 1])
% 
% figure
% for i=1:1600
%     z(i)=round(359*(pi+atan2(xf_sort(i,4),xf_sort(i,3)))/(2*pi))+1;
% %     z(i)
% %     map_rgb(z(i),:)
%     plot(xf_sort(i,3),xf_sort(i,4),'.','markersize',20,'color',map_rgb(z(i),:)), hold on 
%     plot(-xf_sort(i,3),-xf_sort(i,4),'.','markersize',20,'color',map_rgb(z(i),:)), hold on 
%     axis([-30 30 -30 30])
%     pause(0.1)
% end
% axis square
% title('V1 Filters in the frequency domain (TICA)')
% xlabel('f_x (c/deg)'),ylabel('f_y (c/deg)'),
% set(gcf,'color',[1 1 1])
% 