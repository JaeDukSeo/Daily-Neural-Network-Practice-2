close all
ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);
% Barra en vertical
% j=1;
% for i=1:360
%     for k=1:4
%         plot([0 1 ],[j j],'color',map_rgb(i,:)), hold on
%         j=j+1;
%     end
% end
% set(gca,'Xtick',[] ,'Xticklabel',[], ...
%    'Ytick',[0 90*4 180*4 270*4 360*4] ,'Yticklabel',{0 '\pi/2', '\pi','3/4 \pi','2\pi'}...
%    ,'yaxislocation','right','FontSize',30)
% 
% axis([0 1 0 360*4])


% Barra en horitzontal
% figure
% j=1;
% for i=1:360
%  for k=1:4
%         plot([j j],[0 1 ],'color',map_rgb(i,:)), hold on
%         j=j+1;
%     end
% end
% set(gca,'Ytick',[] ,'Yticklabel',[], ...
%    'Xtick',[0 90*4 180*4 270*4 360*4] ,'Xticklabel',{0 '\pi/2', '\pi','3/4 \pi','2\pi'} ...
%    ,'FontSize',30) 
% axis([0 360*4 0 1])

% barretes 
%RECTANGLE('Position', [x y w h])
w=10; %width
h=2; %height
kk=12;
Nbar=14
Vindex=linspace(1, 180,Nbar);
Vangles=linspace(0, 180,Nbar);
V2=flip(Vindex(5:end));
figure
for i=1:floor(Nbar/2)
    
    %corner position
    x=0; y=0; 
    xv=[x x+w x+w x x];
    yv=[y y y+h y+h y];
    hold on
   
    %rotate angle alpha
    alpha=Vangles(i)*2*pi/360;
    R(1,:)=xv;
    R(2,:)=yv;
    XY=[cos(alpha) -sin(alpha);sin(alpha) cos(alpha)]*R;  
   % XY=R;
   
    
      if i<floor(Nbar/2)
          fac=sin(alpha)*(w/2);
          fill(XY(1,:)-kk*i,XY(2,:)-fac,map_rgb(floor(Vindex(i)),:),'LineStyle','none'), hold on
    %fill(XY(1,:),XY(2,:)-fac,map_rgb(floor(Vindex(i)),:),'LineStyle','none'), hold on
    
         axis equal
         % simetria axial
         XY=[-1 0;0 1]*[XY(1,:)-kk*i; XY(2,:)-fac];
        fill(XY(1,:)-Nbar*kk,XY(2,:),map_rgb(floor(V2(i)),:),'LineStyle','none'), hold on
      else
          alpha=90*2*pi/360;
          R(1,:)=xv;
          R(2,:)=yv;
          XY=[cos(alpha) -sin(alpha);sin(alpha) cos(alpha)]*R;  

          fac=sin(alpha)*(w/2);
          fill(XY(1,:)-kk*i,XY(2,:)-fac,map_rgb(90,:),'LineStyle','none'), hold on
         %fill(XY(1,:),XY(2,:)-fac,map_rgb(floor(Vindex(i)),:),'LineStyle','none'), hold on
      end
end

