function [seq,P,ESTT,ACET,TTTT,ESTR,ACER,RRR,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,se_ve,p2dx,p2dy]=pelirott(acel,acela,Eto,Ero,to,At,N,M,Dom,cond,cres,viu,pilum,amb,dom2d,a,b,c,al,bet,gamm,nfac,npix,lim,fig,tra,nom)

%
% NEWTONIAN_SEQUENCE generates the movie (and dynamical data!) of an elipsoid in an acceleration field
%
%   - The initial conditions (at time t0): position-speed (center of mass), and
%     angle-angular-speed (with regard to the center of mass)
%
%   - The expression of the linear acceleration and the angular acceleration.
%     Acceletations are given as functions of the location (x) and the time parameter (p)
%          'a(x,p)'  = linear acceleration
%          'aa(x,p)' = angular acceleration
%
%   - The values for t0 (initial time) and dt (time between frames).
%
% USE: [sec,movie,x_v,a,T,ang_ang_speed,a_ang,R,points0X,points0Y,points0Z,p3dx,p3dy,p3dz,is_vis,p2dx,p2dy] = ...
%       newtonian_sequence('a(x,p)','aa(x,p)',x_v_0,ang_ang_speed,t0,dt,...
%                           N_frames,M_interm,[xm xM ym yM zm zM],box?,coef_restit,[azimut_view elev_view],...
%                           [azimut_illum elev_illum],background,dom_image,a,b,c,alfa,beta,gamma,nfacet,...
%                           npix,limit_alineac,fig);
%
% SEE ILLUSTRATIVE VALUES IN: example_newtonian_sequences.m
%
% Inputs:
%     'a(x,p)'
%     'aa(x,p)'
%     x_v_0
%     ang_ang_speed
%     t0
%     dt
%     N_frames
%     M_interm
%     [xm xM ym yM zm zM]
%     box?
%     coef_restit
%     [azimut_view elev_view]
%     [azimut_illum elev_illum]
%     background (max = 64)
%     dom_image,
%     a,
%     b,
%     c,
%     alfa,
%     beta,
%     gamma,
%     nfacets,...
%     npix,
%     limit_align,
%     fig
%
% Outputs:
%     sec
%     matlab_movie,
%     x_v,
%     a,
%     T,
%     ang_ang_speed,
%     a_ang,
%     R,
%     points0X,points0Y,points0Z,
%     p3dx,p3dy,p3dz,
%     is_vis,
%     p2dx,p2dy
%
% The core functions are:
%    * dinam_tr.m (translation and rotation newtonian dynamics -Runge-Kutta integration-) 
%    * pintael2.m (illumination of facets and projection onto the camera)
%    * elipso3.m  (definition of the facets of an ellipsoid)  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION: FIRST FRAME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i=1
t=to:At:to+At*(N-1);
ESTT=Eto;
ESTR=Ero;

% Aceleraciones en el instante inicial
ACET=funcion(ESTT(1,1:3),to,acel);
ACER=funcion(ESTT(1,1:3),to,acela);

figure(fig),ax,axis('ij'),axis([1 npix 1 npix]);P=moviein(N);

% Estructura del objeto
[Tra0,ptos0]=elipso3(a,b,c,al,bet,gamm,nfac);

np=size(ptos0);np=np(1);
ptos0X=ptos0(:,1)';
ptos0Y=ptos0(:,2)';
ptos0Z=ptos0(:,3)';
ss=size(Tra0);ss=ss(1);

% Posiciones iniciales en el espacio 3d
incT=ones(ss,1)*[ESTT(1,1:3) ESTT(1,1:3) ESTT(1,1:3) ESTT(1,1:3)];
Tra=Tra0+incT;
p=ptos0+ones(np,1)*ESTT(1,1:3);
p3dx(1,:)=p(:,1)';
p3dy(1,:)=p(:,2)';
p3dz(1,:)=p(:,3)';

% Imagen del objeto desde un cierto punto de vista y con cierta iluminacion
[A,TT,p2d,c2,es_veu]=pintael2(p,Tra,viu,amb,pilum,dom2d,npix,0,0,lim);
se_ve(1,:)=es_veu;

% Posiciones iniciales retinianas
p2dx(1,:)=p2d(:,1)';
p2dy(1,:)=p2d(:,2)';

% Imagen inicial
figure(fig),image(A),colormap(gray),ax
P(:,1)=getframe;

try
nombre=[nom,int2str(1),'.fot'];
pun=fopen([tra,nombre],'w');
fwrite(pun,A,'uchar');
fclose(pun);
end

roac=[1 0 0;0 1 0;0 0 1];

%%%%%%%%%%%%%%%%%%%%
% LOOP OVER THE FRAMES
%%%%%%%%%%%%%%%%%%%%

seq = A;

for i=1:N-1
    i+1
    tt=linspace(t(i),t(i+1),M+2);
    dt=At/(M+1);                                       
    E=ESTT(i,:);                                      
    Er=ESTR(i,:);                                      
    RR=[1 0 0;0 1 0;0 0 1];
    R=[1 0 0;0 1 0;0 0 1];
    for j=1:M+1 
        roac=R*roac;
        RR=R*RR;
        % Newtonian Dynamics (displacement and rotation from the field)
        [E,a,T,Er,ar,R]=dinam_tr(ptos0,acel,acela,E,Er,roac,dt,tt(j),cond,Dom,cres);  
    end
    roac=R*roac;
    RR=R*RR;
    ESTT(i+1,:)=E;
    ACET(i+1,:)=a;
    TTTT(i,1:3)=ESTT(i+1,1:3)-ESTT(i,1:3);
    ESTR(i+1,:)=Er;
    ACER(i+1,:)=ar;
    RRR(i,:)=[RR(1,:) RR(2,:) RR(3,:)];
    for l=1:ss
        Tra(l,1:3)=(roac*(Tra0(l,1:3))')';
        Tra(l,4:6)=(roac*(Tra0(l,4:6))')';
        Tra(l,7:9)=(roac*(Tra0(l,7:9))')';
        Tra(l,10:12)=(roac*(Tra0(l,10:12))')';
    end
    incT=ones(ss,1)*[ESTT(i+1,1:3) ESTT(i+1,1:3) ESTT(i+1,1:3) ESTT(i+1,1:3)];
    Tra=Tra+incT;
    for ll=1:np
        p(ll,:)=(roac*(ptos0(ll,:))'+ESTT(i+1,1:3)')';
    end
    % Posiciones 3D en el instante 1+i
    p3dx(i+1,:)=p(:,1)';
    p3dy(i+1,:)=p(:,2)';
    p3dz(i+1,:)=p(:,3)';
    
    % Imagen en el instante 1+i
    [A,TT,p2d,c2,es_veu]=pintael2(p,Tra,viu,amb,pilum,dom2d,npix,0,0,lim);
    se_ve(i+1,:)=es_veu;
    
    % Retinal 2d positions
    p2dx(i+1,:)=p2d(:,1)';
    p2dy(i+1,:)=p2d(:,2)';
    figure(fig),image(A),colormap(gray),ax
    P(:,i+1)=getframe;
    
    seq = [seq A];
    
    try
    nombre=[nom,int2str(i+1),'.fot'];
    pun=fopen([tra,nombre],'w');
    fwrite(pun,A,'uchar');
    fclose(pun);
    end
end

try
eval(['save ',tra,nom,'.pel P'])
eval(['save ',tra,nom,'.dat ESTT ACET TTTT ESTR ACER RRR ptos0X ptos0Y ptos0Z p3dx p3dy p3dz se_ve p2dx p2dy'])
end

figure(fig+1),plot3(ESTT(:,1),ESTT(:,2),ESTT(:,3),'y.');xlabel('X');ylabel('Y');zlabel('Z');axis(Dom),view(viu),ax