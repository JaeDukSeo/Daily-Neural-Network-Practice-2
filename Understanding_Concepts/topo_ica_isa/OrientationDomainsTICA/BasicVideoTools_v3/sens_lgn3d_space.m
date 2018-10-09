function [G,Ge,Gi]=sens_lgn3d_space(Nx,Ny,Nt,fsx,fsy,fst,xop,yop,top,order_p,sigmax_p,sigmat_p,xon,yon,ton,order_n,sigmax_n,sigmat_n,excit_vs_inhib)

% SENS_LGN3D_SPACE computes the (achromatic) receptive field of LGN cells  
% in the spatio-temporal domain.
% These cells are mixtures of excitatory and inhibitory Gaussian regions that 
% respond monophasically or biphasically in time [Cai, Freeman J. Neurophysiol. 1997, 
% J. Neurosci. 2003, Devalois 2003]:
%
%        G(x,t) = G_excit(x)*G_excit(t) - G_inhib(x)*G_inhib(t)
%
% where:
%
%        G_i(x) = A_i*exp(-(x-x_i0).^2/sigma_x_i^2)
%
%        G_i(t) = B_i*gaussian_derivative_1d(t-t_i0,sigma_t_i,order) 
%                   order = 0 for monophasic (Gaussian)
%                   order = 1 for biphasic (First derivative of Gaussian)
%
%        NOTE!: Freeman et al use a difference of gamma functions instead of
%               gaussian derivatives to model the temporal oscillations.
%               This is something to be improved here!
% 
% The amplitudes B_i are (automatically) selected so that norm(G_i(t)) = 1. 
% In this way, the relative amplitude of the excitatory and inhibitory
% parts is given by the A_i.
% The amplitudes A_i are selected so that the proportion of energies of the 
% excitatory and inhibitory parts (excit_over_inhib) is the one defined by 
% the user.
% Finally the global receptive field is normalized to have unit norm.
% Nevertheless, note that both parts are separatedly given so that the user
% can apply other relative normalizations.
%
% Here one can manipulate the spatio-temporal location, width and energy of 
% the excitatory and inhibitory Gaussians separatedly: variables X_p
% (positive) and X_n (negative).
% 
% In order to select the temporal location and width of the temporal impulses, 
% note that the different orders of the derivative of a Gaussian are shifted 
% in time (the biphasic starts sooner) and the effective width is different 
% (biphasic lasts longer for the same sigma). 
% Please play with the example in the help of gaussian_derivative_1d.m.
%
% The routine requires the sizes of the discrete domain (rows,columns,frames) 
% and the corresponding sampling frequencies (fy,fx,ft)
%
% SYNTAX: [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib)
% 
% Example 3D: 
%     
%           columns_x = 40;
%           rows_y = 40;
%           frames_t = 24;
%           fsx = 40;
%           fsy = 40;
%           fst = 24;
%           xo_p = 0.5;
%           yo_p = 0.5;
%           to_p = 0.3; 
%           order_p = 1;
%           sigmax_p = 0.05; 
%           sigmat_p = 0.1; 
%           xo_n = 0.5;
%           yo_n = 0.5;
%           to_n = 0.3; 
%           order_n = 1;
%           sigmax_n = 0.2; 
%           sigmat_n = 0.1;
%           excit_vs_inhib = 1;
%
% [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
%
% Example 2D: 
%     
%           columns_x = 40;
%           rows_y = 40;
%           frames_t = 1;      % Only one frame! 
%           fsx = 40;
%           fsy = 40;
%           fst = 24;
%           xo_p = 0.5;
%           yo_p = 0.5;
%           to_p = 0.3; 
%           order_p = 1;
%           sigmax_p = 0.05; 
%           sigmat_p = 0.1; 
%           xo_n = 0.5;
%           yo_n = 0.5;
%           to_n = 0.3; 
%           order_n = 1;
%           sigmax_n = 0.2; 
%           sigmat_n = 0.1;
%           excit_vs_inhib = 1;
%
% [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
%
% Example 1D: take a slice at y = y0 at the previous G. 
%     
%           G1d = G(round(yo_p*fsy),:);
%           G1d = G1d/norm(G1d);
%


% [G,Ge,Gi]=sens_lgn3d_space(Nx,Ny,Nt,fsx,fsy,fst,xop,yop,top,order_p,sigmax_p,sigmat_p,xon,yon,ton,sigmax_n,sigmat_n,excit_vs_inhib)

[gtp,tt] = gaussian_derivative_1d(fst,Nt,top,sigmat_p,order_p);
gtp = gtp/norm(gtp);
[gtn,tt] = gaussian_derivative_1d(fst,Nt,ton,sigmat_n,order_n);
gtn = gtn/norm(gtn);

[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(Ny,Nx,Nt,fsx,fsy,fst);

Gtp = ones(Ny,Nx,Nt);
Gtn = ones(Ny,Nx,Nt);
for i=1:Nt
    Gtp(:,:,i) = gtp(i)*Gtp(:,:,i);
    Gtn(:,:,i) = gtn(i)*Gtn(:,:,i);
end
Gtp = now2then(Gtp);
Gtn = now2then(Gtn);

Gp = exp(-(x-xop).^2/sigmax_p^2-(y-yop).^2/sigmax_p^2);
Gp = Gp/sqrt(sum(Gp(:).^2));
Gn = exp(-(x-xon).^2/sigmax_n^2-(y-yon).^2/sigmax_n^2);
Gn = Gn/sqrt(sum(Gn(:).^2));

Ge = Gp.*Gtp;
Gi = Gn.*Gtn;

Ge = Ge/sqrt(sum(Ge(:).^2));
Gi = Gi/sqrt(sum(Gi(:).^2));

Ge = excit_vs_inhib*Ge;

G = Ge - Gi; 

G = G/sqrt(sum(G(:).^2));
