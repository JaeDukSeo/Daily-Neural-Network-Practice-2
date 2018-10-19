function save_avi_file(sec,m,M,col,file,fig)

file
% Prepare the new file.
    vidObj = VideoWriter(file);
    open(vidObj);
 
s=size(sec);

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
       currFrame = getframe;
       writeVideo(vidObj,currFrame);        
        
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
       currFrame = getframe;
       writeVideo(vidObj,currFrame);        

    end   
   
end

    % Close the file.
    close(vidObj);
    
    