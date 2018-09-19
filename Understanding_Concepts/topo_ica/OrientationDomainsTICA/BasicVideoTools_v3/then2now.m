function x_3d = then2now(x_2d,nc);

%
% THEN2NOW transforms movie data from the (old-fashioned) 2d array format to the 
% (natural) 3d array format.
%
% The natural way to store a tristimulus component of a movie sequence of 
% F frames of R rows and C columns is a 3D array of sizes R*C*F.
% 
% Nevertheless, 20th century versions of Matlab only allowed for 2D arrays.
% Therefore I developed a (counter intuitive) 2D way to store movies: 
% stack one frame after the other: sequence = [F1 F2 F3 ... FF], where Fi 
% is an R*C matrix that contains the values of the i-th frame, and sequence
% is an R*(C*F) matrix.
%
% I developed a set of functions associated to that format in order to:
%  (1) compute and visualize 3D fourier transforms. fft3, ifft3, show_fft3
%  (2) define functions in the 3D fourier domain and in the 3D spatio-temporal domain.
%            spatio_temp_freq_domain  
%  (3) generate movie structures from data. build_achrom_movie
%  (4) get/set individual frames from a sequence (i.e. the equivalent to Y(:,:,i)), 
%  (5) get/set "temporal lines" (i.e. the equivalent to Y(i,j,:)).
%
% In order to keep using the useful functions (1-3) it is convenient to 
% transform from the natural format to the old-fashioned format (and come 
% back to the natual format).
%
% SYNTAX: Y_3d = then2now(Y_2d,nc);
%
%    Y_2d = movie data in 2d array format
%     nc  = number of columns in each frame
%    Y_3d = movie data in 3d array format
%
% See now2then for the forward re-arrangement.
%

[nr,nc_t_nf] = size(x_2d);

nf = nc_t_nf/nc;

x_3d = zeros(nr,nc,nf);

for i = 1:nf
    x_3d(:,:,i) = sacafot(x_2d,nr,nc,i);
end