function [M,BBB] = disp_spatio_temp_patches(B,N,Nx,Nf,buf,fig)

% DISP_SPATIO_TEMP_PATCHES generates a joint movie of N*N spatio-temporal patches
% in the columns of the matrix B. 
% The spatio-temporal patches in the columns of B are arranged according to
% im2colcube.m / col2imcube.m 
% 
% This function extracts each frame (or band) of the corresponding patch
% and computes the global frame (or band) by using disp_patches.m
% As in disp_patches, this function assumes that the size is equal in both 
% spatial dimensions (squared patches).
%
% [movie,array] = disp_spatio_temp_patches(B,N,Nx,Nf,buf,fig)
%
%      B = matrix with N*N columns with the patches
%      N = number of rows/cols in the array of functions
%      Nx = spatial size of the patches 
%      Nf = number of frames (bands) in the patches
%      buf = space between patches
%      

M = moviein(Nf);

for fot = 1:Nf
    
    if fot ==1
        BB = zeros(Nx*Nx,Nf);
        for i = 1:N*N
            base = col2imcube(B(:,i),[Nx Nf],[Nx Nx]);
            im_base = base(:,:,fot);
            BB(:,i) = im_base(:);
        end
        
        array = disp_patches(BB,N,buf);
        
        figure(fig),colormap gray,imagesc(array),axis square,axis off
        M(fot) = getframe;
        
        s = size(array);
        BBB = zeros(s(1),s(2),Nf);
        BBB(:,:,1) = array;
        
    else
        BB = zeros(Nx*Nx,Nf);
        for i = 1:N*N
            base = col2imcube(B(:,i),[Nx Nf],[Nx Nx]);
            im_base = base(:,:,fot);
            BB(:,i) = im_base(:);
        end
        
        array = disp_patches(BB,N,buf);
        
        figure(fig),colormap gray,imagesc(array),axis square,axis off
        M(fot) = getframe;
        
        BBB(:,:,fot) = array;
        
    end
    
end
