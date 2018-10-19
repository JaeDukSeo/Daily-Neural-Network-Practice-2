function [Eimp,Timp,Erimp,Rimp,ddt]=rebota2(ptos0,Rac,E0,E0r,acel,acela,dt,t,limit,coef)

% REBOTE calcula la posicion y velocidad de salida de una particula
% tras un rebote (con perdida de energia) con las paredes de una caja.
%
% El resultado del choque es la inversion de la velocidad normal a la 
% superficie de la caja y una disminucion de la energia cinetica en un
% factor dado por el coeficiente de restitucion [0,1].
% 
% Se supone que mientras tiene lugar el rebote la aceleracion que ac-
% tua sobre el cuerpo permanece constante.
%
% La caja es un paralelepipedo definido por sus extremos.
% 
% USO: [Eimp,Timp,Erimp,Rimp,ddt]=rebota2(posicion-velocidad(t),'a(x,p)',dt,t,limites_caja,coefic_restit);

lim1=limit(1:2);
lim2=limit(3:4);
lim3=limit(5:6);

[a,l1]=min(abs([E0(1) E0(1)]-lim1));l1=lim1(l1);
[a,l2]=min(abs([E0(2) E0(2)]-lim2));l2=lim2(l2);
[a,l3]=min(abs([E0(3) E0(3)]-lim3));l3=lim3(l3);

con=(E0(4:6)==0);
E00=E0;
ccc=0;
cc=0;
if sum(con)>0
    [a,coo1]=max(con);
    vm=max(abs(E0(4:6)));
    E00(3+coo1)=vm;
    cc=1;
end
con=(E00(4:6)==0);
E000=E00;
if sum(con)>0
    [a,coo2]=max(con);
    vm=max(abs(E00(4:6)));
    E000(3+coo2)=vm;
    ccc=1;
end

t_imp=([l1 l2 l3]-E0(1:3))./E000(4:6);
t_neg=(t_imp<=0)*2*dt;
t_imp=t_imp.*(t_imp>0)+t_neg;
tm=max(t_imp);
if cc>0
  t_imp(coo1)=2*tm;
end
if ccc>0
  t_imp(coo2)=2*tm;
end

[t_imp,coord]=min(t_imp);

[Eimp,aimp,Timp]=dinamica(acel,E0,t_imp,t);
[Erimp,arimp,Rimp]=dinamrot(acela,E0r,t_imp,t);

velo=Eimp(4:6);
Eimp(3+coord)=-Eimp(3+coord);
Eimp(4:6)=sqrt(coef)*Eimp(4:6);

ddt=(dt-t_imp);

Np=size(ptos0);Np=Np(1);
for i=1:Np
    ptos(i,:)=(Rimp*Rac*(ptos0(i,:))'+Eimp(1:3)')';
end

if (limit(2*coord)-Eimp(coord))>(-limit(2*coord-1)+Eimp(coord))
    [mini,pos]=min(ptos(:,coord));
    pto_c1=ptos(pos,:);     
else
    [mini,pos]=max(ptos(:,coord));
    pto_c1=ptos(pos,:);
end
cen=Eimp(1:3);
ptos(pos,:)=cen;

if (limit(2*coord)-Eimp(coord))>(-limit(2*coord-1)+Eimp(coord))
    [mini,pos]=min(ptos(:,coord));
    pto_c2=ptos(pos,:);     
else
    [mini,pos]=max(ptos(:,coord));
    pto_c2=ptos(pos,:);
end
ptos(pos,:)=cen;

if (limit(2*coord)-Eimp(coord))>(-limit(2*coord-1)+Eimp(coord))
    [mini,pos]=min(ptos(:,coord));
    pto_c3=ptos(pos,:);     
else
    [mini,pos]=max(ptos(:,coord));
    pto_c3=ptos(pos,:);
end
ptos(pos,:)=cen;

if (limit(2*coord)-Eimp(coord))>(-limit(2*coord-1)+Eimp(coord))
    [mini,pos]=min(ptos(:,coord));
    pto_c4=ptos(pos,:);     
else
    [mini,pos]=max(ptos(:,coord));
    pto_c4=ptos(pos,:);
end
ptos(pos,:)=cen;

if (pto_c1(coord)==pto_c2(coord))&(pto_c1(coord)==pto_c3(coord))&(pto_c1(coord)==pto_c4(coord))
      pto_c=(pto_c1+pto_c2+pto_c3+pto_c4)/4;
elseif (pto_c1(coord)==pto_c2(coord))&(pto_c1(coord)==pto_c3(coord))&(pto_c1(coord)~=pto_c4(coord))
      pto_c=(pto_c1+pto_c2+pto_c3)/3;
elseif (pto_c1(coord)==pto_c2(coord))&(pto_c1(coord)~=pto_c3(coord))
      pto_c=(pto_c1+pto_c2)/2;
else
      pto_c=pto_c1;
end

pto_c;
r=pto_c-cen
velo
mr=sqrt(r*r');

Erimp(4:6)=sqrt(coef)*(-(1/mr)*p_vect(r/mr,velo)+Erimp(4:6));
