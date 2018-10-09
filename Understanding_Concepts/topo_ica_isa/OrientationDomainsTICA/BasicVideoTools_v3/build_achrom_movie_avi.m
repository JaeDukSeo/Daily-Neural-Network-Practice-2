function P = build_achrom_movie_avi(sec,m,M,col,fig,fps,folder,name)

% BUILD_ACHROM_MOVIE_AVI generates both a Matlab movie and an *.avi file from 2d or 3d data.
% This function displays the movie in figure "fig" and scales the
% values of the colormap so that "m" is black and "M" is white.
% BUILD_ACHROM_MOVIE does the same without savin the file.
%
% SYNTAX: MOV = build_achrom_movie_avi(Y,m,M,num_column,fig,folder,name);
%
%        Y = movie data (2d or 3d array)
%        m = value to be depicted as black
%        M = value to be depicted as white
%        num_columns = number of columns in each frame
%        fig = figure where the movie will be displayed
%        fps = Temporal sampling frequency (frames per second)
%        folder = string with full path to the folder where the avi file will be stored 
%        name = string name of the *.avi file (extension is automatically appended).
%               note that the file will be [folder,name,'.avi'] so write
%               the folder accordingly (ending with / or \)
%
%        MOV = Matlab movie structure that can be displayed using "movie" or "implay"
%          
%        NOTE that implay also plays 3D arrays so that the movie structure
%        is less necessary than last century. Given the evolution of the
%        movie function (and movie2avi) since 2010, it seems matworks plans
%        to remove the movie structure (which always seemed kind of
%        arbitrary).
%

s=size(sec)

    % Prepare the new file.
    vidObj = VideoWriter([folder,name,'.avi'],'Uncompressed AVI');
    vidObj.FrameRate=fps;
    open(vidObj);

if length(s)==2

   nf=s(2)/col;
   nr = s(1);
   nc = col;
   
   % m = min(min(sec));
   % M = max(max(sec));

    P = moviein(nf);

    %sec = 64*(sec-m)/(M-m);

    for i=1:nf
        im=sacafot(sec,nr,nc,i);
        figure(fig);colormap gray;
        imagesc(im,[m M]),axis('equal'),axis('off')
        lala=getframe;
        P(:,i)=lala;
        writeVideo(vidObj,lala);
        
    end   
   
else
   
   nr = s(1);
   nc = col
   nf = s(3); 

   % m = min(min(min(sec)));
   % M = max(max(max(sec)));

    P = moviein(nf);

    %sec = 64*(sec-m)/(M-m);

    for i=1:nf
        im=sec(:,:,i);
        figure(fig);colormap gray;
        imagesc(im,[m M]),axis('equal'),axis('off')
        lala=getframe;
        P(:,i)=lala;
        writeVideo(vidObj,lala);
    end   
   
end

% Close the file.
    close(vidObj);
