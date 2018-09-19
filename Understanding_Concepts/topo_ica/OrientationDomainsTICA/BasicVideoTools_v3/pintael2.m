function [A,TT,p2d,c2,seve]=pintael2(ptos,tra,viu,amb,pilu,dom,N,fig,pin,li)

% PINTAEL2 generates a no-perspective projection of an ellipsoid defined by 
% the polygons TRA (given by elipso2.m) onto an image plane orthogonal to the 
% optical axis and view point defined by [azimut_v elevation_v].
%
% This routine removes hidden lines and computes the shadow of each facet assuming
% lambertian reflection and certain ambient light. 
% It is assumed that illumination is done in certain illumination direction 
% defined by [azimut_ilum elevation_ilum].
%
% USE: [imag,TT,p2d,col,seve]=pintael2(elips_points,trapezoids,[azimut_v elevacion_v],ambient_light,[azimut_ilum elevacion_ilum],...
%                                      dom2d,N_pixels,fig,pinta?,li);
%
%      See its use in newtonian_sequence.m with representative values given
%      in examples_newtonian_sequences.m
% 

np=size(ptos);np=np(1);
seve=zeros(1,np);

zv=sin(viu(2)*pi/180);
xy=cos(viu(2)*pi/180);
xv=xy*cos(viu(1)*pi/180);
yv=xy*sin(viu(1)*pi/180);
v=-[-yv xv -zv];
v=v/sqrt(sum(v.^2));

zi=sin(pilu(2)*pi/180);
xy=cos(pilu(2)*pi/180);
xi=xy*cos(pilu(1)*pi/180);
yi=xy*sin(pilu(1)*pi/180);
vi=-[-yi xi -zi];
vi=vi/sqrt(sum(vi.^2));

A=zeros(N,N);

x=linspace(dom(1),dom(2),N);
x=ones(N,1)*x;
y=linspace(dom(3),dom(4),N);
y=y(length(y):-1:1);
y=y'*ones(1,N);

s2=size(tra);
s2=s2(1);
k=1;
for i=1:s2
    P1=(tra(i,1:3))';
    P2=(tra(i,4:6))';      
    P3=(tra(i,7:9))';
    P4=(tra(i,10:12))';
    a=P2-P1;
    b=P3-P2;
    c=P4-P3;
    d=P1-P4;
    if (any((d==-a)==0))     % ES DISTINTO?
       pv=p_vect(a,b);
       s1=-pv/sqrt(sum(pv.^2));
%       pv=p_vect(b,c);
%       s2=-pv/sqrt(sum(pv.^2));
%       pv=p_vect(c,d);
%       s3=-pv/sqrt(sum(pv.^2));
%       pv=p_vect(d,a);
%       s4=-pv/sqrt(sum(pv.^2));
%       if (s1*v'<0)&(s2*v'<0)&(s3*v'<0)&(s4*v'<0)
       if s1*v'<0
          [x2d,aa]=proyec(viu,0,[0 0 0],[P1';P2';P3';P4']);
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P1')')==0);
          seve(pos)=1;
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P2')')==0);
          seve(pos)=1;
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P3')')==0);
          seve(pos)=1;
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P4')')==0);
          seve(pos)=1;
          ctet=vi*s1';
          if ctet<0
               c2(k)=floor(-(64-amb)*ctet)+amb;
          else
               c2(k)=amb;
          end
          TT(k,:)=[x2d(1,:) x2d(2,:) x2d(3,:) x2d(4,:)];
          A=pintapo2(A,x,y,[x2d(1,:);x2d(2,:);x2d(3,:);x2d(4,:)],c2(k),li);
          if pin>0 
          figure(fig),image(A),colormap(gray),ax
          end
          k=k+1;
       end
    else 
       a=P2-P1;
       b=P3-P2;
       c=P1-P3;
       pv=p_vect(a,b);
       s1=-pv/sqrt(sum(pv.^2));
%       pv=p_vect(b,c);
%       s2=-pv/sqrt(sum(pv.^2));
%       pv=p_vect(c,d);
%       s3=-pv/sqrt(sum(pv.^2));
       if (s1*v'<0)
          [x2d,aa]=proyec(viu,0,[0 0 0],[P1';P2';P3';P4']);
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P1')')==0);
          seve(pos)=1;
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P2')')==0);
          seve(pos)=1;
          [m,pos]=max(sum(abs(ptos-ones(np,1)*P3')')==0);
          seve(pos)=1;
          ctet=vi*s1';
          if ctet<0
               c2(k)=floor(-(64-amb)*ctet)+amb;
          else
               c2(k)=amb;
          end
          TT(k,:)=[x2d(1,:) x2d(2,:) x2d(3,:) x2d(4,:)];
          A=pintapo2(A,x,y,[x2d(1,:);x2d(2,:);x2d(3,:);x2d(4,:)],c2(k),li);
          if pin>0 
          figure(fig),image(A),colormap(gray),ax
          end
          k=k+1;
       end
    end
end
[p2d,aa]=proyec(viu,0,[0 0 0],ptos);