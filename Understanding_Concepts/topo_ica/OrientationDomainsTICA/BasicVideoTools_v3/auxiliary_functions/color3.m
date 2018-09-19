function [c,a]=color3(x,y,w,h,pos,lad,col,i)
% Calcula el color mitja en un rectangle en funcio dels colors dels
% objectes rectangulars en pos i lad a partir de pos(i,:) lad(i,:)
%
c=0;
a=w*h;
ww=0;
for j=i:size(pos,1)
   [xx,yy,ww,hh]=intersec( [x,y],[w,h],pos(j,:),lad(j,:) );
   if ww>0
      %c0=col(j);
      ac=col(j)*ww*hh; %% acumulador
      if (xx>x) 
         [c1,a1]=color3(x,y,xx-x,h,pos,lad,col,j+1);
         ac=ac+c1*a1;
      %else
      %   c1=0;a1=0;
      end
      if (x+w>xx+ww)
         [c2,a2]=color3(xx+ww,y,x+w-xx-ww,h,pos,lad,col,j+1);
         ac=ac+c2*a2;
      %else
      %   c2=0;a2=0;
      end
      if (yy>y)
         [c3,a3]=color3(xx,y,ww,yy-y,pos,lad,col,j+1);
         ac=ac+c3*a3;
      %else
      %   c3=0;a3=0;
      end
      if (y+h>yy+hh)
         [c4,a4]=color3(xx,yy+hh,ww,y+h-yy-hh,pos,lad,col,j+1);
         ac=ac+c4*a4;   
      %else
      %   c4=0;a4=0;
      end
      
      %c=(c0*ww*hh + c1*a1 + c2*a2 + c3*a3 + c4*a4 )/w/h;
      c=ac/a;
      break;
   end
end

