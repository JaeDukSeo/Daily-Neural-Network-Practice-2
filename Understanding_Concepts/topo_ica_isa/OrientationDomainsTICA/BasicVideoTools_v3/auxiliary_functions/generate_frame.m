function I=generate_frame(pos,lad,col,f,c,fs,dib);

%
% GENRATE_FRAME generates an image from location, size and color data of N rectangles.
%
% We assume (1) opaque objects (opaque rectangles), (2) later objects occlude
% previous objects in the list, and (3) the discrete sensors (pixels) average 
% the energy of distinct objects falling in the same pixel.
%
% This function is intended to be used with dots_sequence.m and a proper
% initialization (see example_random_dots_sequence.m)
%
% USE:  I = generate_image(locations,sizes,colors,N_rows,N_cols, fs, represent?)
% 
% Input:
%     locations = N*2 matrix with the 2d locations (in deg) of the N objects
%         sizes = N*2 matrix with the [width height] (in deg) of the N rectangles 
%        colors = N*1 vector with the gray level of each object 
%        N_rows = number of rows in the image
%        N_cols = number of columns in the image 
%            fs = sampling frequency (in cpd) 
%    represent? = 0 (no representation) or 1 (represent frame and overlapping rectangles)
%
% Output:
%    I = the discrete image (N_rows*N_cols matrix)
%
% See example on how to set the input parameters in example_random_dots_sequence.m
% 


YY=f/fs;
XX=c/fs;

incrx=XX/c;
incry=YY/f;

I=zeros(f,c);


x=linspace(0,XX-incrx,c);
y=linspace(0,YY-incry,f);


for fil=1:f
   %%y=  YY-fil*incry;
   for colu=1:c
      %%x=  (colu-1)*incrx;
     
      indices1=( pos(:,1)<(x(colu)+incrx ) );
      indices2=( pos(:,2)<(y(fil)+incry  ) );
       
      indices3=( (pos(:,1)+lad(:,1))>(x(colu) ) );
      indices4=( (pos(:,2)+lad(:,2))>(y(fil) ) );
      
      indices=indices1 & indices2 & indices3 & indices4;

      %indices=ones(size(pos,1),1);
            
      posi=pos(logical(indices),:);
      ladi=lad(logical(indices),:);
      coli=col(logical(indices),:);
      

      if (isempty(posi))
        co=0;
      else
         [co,ar]=color3(x(colu),y(fil),incrx,incry,posi,ladi,coli,1);
      end
      I(fil,colu)=co;
     
   end
end


if (dib==1)
  figure(1);
  pintacua(pos,lad,XX,YY,incrx,incry)
  figure(2);
  colormap(gray);
  imagesc(I);
end  

