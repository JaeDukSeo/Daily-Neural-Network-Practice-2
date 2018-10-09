function [Tra,ptos]=elipso3(a,b,c,alfa,beta,gama,N);

%
% ELIPSO3 computes the polygons of a faceted ellipsoid.
% The polygons (4 sides trapezoids) are given in this way:
%
%      trapezoids(i,:)=[x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
%
% USE: [trapezoids,points]=elipso3(a,b,c,alfa,beta,gama,N_facets);
%

alfa=alfa*pi/180;
beta=beta*pi/180;
gama=gama*pi/180;

aa=max([a b c]);
x=linspace(-1.1*aa,1.1*aa,N);
y=x'*ones(1,N);
x=y';

zM=sqrt((a*b*c)^2*ones(N,N)-(b*c)^2*x.^2-(a*c)^2*y.^2)/(a*b);

c1=(imag(zM)==0);

zM=c1.*zM;

k=1;
for i=1:N
    for j=1:N
        if zM(i,j)>0 
           ptos(k,:)=[x(i,j) y(i,j) zM(i,j)];
           k=k+1;
        end
    end
end

ptos=[ptos;ptos(:,1:2) -ptos(:,3)];

k=1;
c1=zeros(N,N);
for i=1:N-1
    for j=1:N-1
        if (zM(i,j)~=0)&(zM(i+1,j)~=0)&(zM(i+1,j+1)~=0)&(zM(i,j+1)~=0)
           Tra(k,:)=[x(i,j) y(i,j) zM(i,j) x(i,j+1) y(i,j+1) zM(i,j+1) x(i+1,j+1) y(i+1,j+1) zM(i+1,j+1) x(i+1,j) y(i+1,j) zM(i+1,j)];
           c1(i,j)=c1(i,j)+1;c1(i+1,j)=c1(i+1,j)+1;c1(i+1,j+1)=c1(i+1,j+1)+1;c1(i,j+1)=c1(i,j+1)+1;
           k=k+1;
        end
    end
end 
Traa=[Tra(:,1) Tra(:,2) -Tra(:,3) Tra(:,10) Tra(:,11) -Tra(:,12) Tra(:,7) Tra(:,8) -Tra(:,9) Tra(:,4) Tra(:,5) -Tra(:,6)];
Tra=[Tra;Traa];
cond=((c1>0)&(c1<4));
[I,J]=find(cond);
NN=length(I);
d=0;
k=I(1);
m=J(1);
tre=zeros(1,12);
l=1;
[kkk,lll,ddd]=contono2(k,m,d,cond);
k(2)=kkk;
m(2)=lll;
d(2)=ddd;
Tre(1,:)=[x(k(1),m(1)) y(k(1),m(1)) zM(k(1),m(1)) x(k(1),m(1)) y(k(1),m(1)) -zM(k(1),m(1)) x(k(2),m(2)) y(k(2),m(2)) -zM(k(2),m(2)) x(k(2),m(2)) y(k(2),m(2)) zM(k(2),m(2))];
for i=2:NN+3
    kk=k(i);
    ll=m(i);
    dd=d(i);
    [kkk,lll,ddd]=contono2(kk,ll,dd,cond);
    k(i+1)=kkk;
    m(i+1)=lll;
    d(i+1)=ddd;
    if (any((d(i-1:i+1)==[1 4 4])==0))&(any((d(i-1:i+1)==[1 4 1])==0))&(any((d(i-1:i+1)==[3 2 2])==0))&(any((d(i-1:i+1)==[3 2 3])==0))&(any((d(i-1:i+1)==[4 3 4])==0))&(any((d(i-1:i+1)==[4 3 3])==0))&(any((d(i-1:i+1)==[2 1 2])==0))&(any((d(i-1:i+1)==[2 1 1])==0))
       Tre(l+1,:)=[x(k(i),m(i)) y(k(i),m(i)) zM(k(i),m(i)) x(k(i),m(i)) y(k(i),m(i)) -zM(k(i),m(i)) x(k(i+1),m(i+1)) y(k(i+1),m(i+1)) -zM(k(i+1),m(i+1)) x(k(i+1),m(i+1)) y(k(i+1),m(i+1)) zM(k(i+1),m(i+1))];
       l=l+1;
    else
       po1=[Tre(l-1,4:6) Tre(l-1,7:9) Tre(l,7:9) Tre(l-1,7:9)];
       po2=[Tre(l-1,1:3) Tre(l-1,4:6) Tre(l,7:9) Tre(l,10:12)];
       po3=[Tre(l-1,10:12) Tre(l-1,1:3) Tre(l,10:12) Tre(l-1,1:3)];
       Tre(l+2,:)=[x(k(i),m(i)) y(k(i),m(i)) zM(k(i),m(i)) x(k(i),m(i)) y(k(i),m(i)) -zM(k(i),m(i)) x(k(i+1),m(i+1)) y(k(i+1),m(i+1)) -zM(k(i+1),m(i+1)) x(k(i+1),m(i+1)) y(k(i+1),m(i+1)) zM(k(i+1),m(i+1))];
       Tre(l-1,:)=po1;
       Tre(l,:)=po2;
       Tre(l+1,:)=po3;
       l=l+2; 
    end
end
kkk=size(Tre);
kkk=kkk(1);
Tra=[Tre(2:kkk-2,:);Tra];

Rz=[cos(alfa) sin(alfa) 0;-sin(alfa) cos(alfa) 0;0 0 1];
Ry=[cos(beta) 0 sin(beta);0 1 0;-sin(beta) 0 cos(beta)];
Rx=[1 0 0;0 cos(gama) sin(gama);0 -sin(gama) cos(gama)];

R=Rx*Ry*Rz;

s=size(Tra);
s1=s(1);
for i=1:s1
       P1=R*(Tra(i,1:3))'; 
       P2=R*(Tra(i,4:6))';
       P3=R*(Tra(i,7:9))'; 
       P4=R*(Tra(i,10:12))'; 
       Tra(i,:)=[P1' P2' P3' P4'];
end

s=size(ptos);
s1=s(1);
for i=1:s1
       ptos(i,:)=(R*(ptos(i,:))')';
end
