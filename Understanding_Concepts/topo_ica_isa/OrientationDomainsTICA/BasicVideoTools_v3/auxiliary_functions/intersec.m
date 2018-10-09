function [x,y,w,h]=intersec(xy1,wh1,xy2,wh2)
% interseccio de dos rectangles qualssevol

x1=xy1(1,1);
y1=xy1(1,2);
w1=wh1(1,1);
h1=wh1(1,2);

x2=xy2(1,1); y2=xy2(1,2);
w2=wh2(1,1); h2=wh2(1,2);

y=y1;
w=0;
h=0;
x=max(x1,x2);
xx=min(x1+w1,x2+w2);
if ( xx>x )
   y=max(y1,y2);
   yy=min(y1+h1,y2+h2);
   if ( yy>y )
      w=xx-x;
      h=yy-y;
   end
end
