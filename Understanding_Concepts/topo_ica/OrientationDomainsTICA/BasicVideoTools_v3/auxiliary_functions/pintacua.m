


function pintacua(pos,lad,XX,YY,incrx,incry)

x1=pos(:,1); xw=pos(:,1)+lad(:,1);
y1=pos(:,2); yh=pos(:,2)+lad(:,2);
A=[x1,xw,xw,x1,x1];
B=[y1,y1,yh,yh,y1];

[n,m]=size(pos);

newplot; hold on;
for i=1:n,
   plot( A(i,:), B(i,:) );
end
   
   axis([0,XX,0,YY]);
   
   set(gca,'YDir','reverse');   
   set(gca,'YTick',[0:incry:YY]);
   set(gca,'XTick',[0:incrx:XX]);
   grid on;
hold off;
   
%set(gca,'YDir','reverse');   
   
%newplot;
%hold on;
%%plot ([x1,x1+w1,x1+w1,x1,x1],[y1,y1,y1+h1,y1+h1,y1]);
%plot ([x2,x2+w2,x2+w2,x2,x2],[y2,y2,y2+h2,y2+h2,y2]);
%plot ([x,x+w,x+w,x,x],[y,y,y+h,y+h,y],'r');
%axis([0,10,0,10]);
%hold off;
