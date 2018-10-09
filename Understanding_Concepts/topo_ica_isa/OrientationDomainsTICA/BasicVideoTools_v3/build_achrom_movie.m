function P = build_achrom_movie(sec,m,M,col,fig)

% BUILD_ACHROM_MOVIE generates a Matlab movie structure from 2d or 3d data.
% This function displays the movie in figure "fig" and scales the
% values of the colormap so that "m" is black and "M" is white.
%
% SYNTAX: M = build_achrom_movie(Y,m,M,num_column,fig);
%
%        Y = movie data (2d or 3d array)
%        m = value to be depicted as black
%        M = value to be depicted as white
%        num_columns = number of columns in each frame
%        fig = figure where the movie will be displayed
%
%        M = Matlab movie structure that can be displayed using "movie" 
%

s=size(sec)

if length(s)==2

   nf=s(2)/col;
   nr = s(1);
   nc = col;
   
   % m = min(min(sec));
   % M = max(max(sec));

    P = moviein(nf);

    sec = 64*(sec-m)/(M-m);

    for i=1:nf
        im=sacafot(sec,nr,nc,i);
        figure(fig);colormap gray;
        image(im),axis('equal'),axis('off')
        P(:,i)=getframe;
    end   
   
else
   
   nr = s(1);
   nc = col
   nf = s(3); 

   % m = min(min(min(sec)));
   % M = max(max(max(sec)));

    P = moviein(nf);

    sec = 64*(sec-m)/(M-m);

    for i=1:nf
        im=sec(:,:,i);
        figure(fig);colormap gray;
        image(im),axis('equal'),axis('off')
        P(:,i)=getframe;
    end   
   
end

