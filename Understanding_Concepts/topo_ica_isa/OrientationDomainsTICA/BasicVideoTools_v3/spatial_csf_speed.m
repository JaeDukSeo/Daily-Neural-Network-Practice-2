function [CSFet,csf_fx_v,fxx,v]=spatial_csf_speed(fsx,fsy,Nx,Ny,v_range,Nv,track,estab);

%
% SPATIAL_CSF_SPEED computes the spatial CSF at different speeds from the spatio-temporal CSF. 
% The user can select motion tracking (as in Daly 98) or fixed gaze with or without eye stabilization (as in Kelly 79).
%
% The program computes the CSF for each speed by computing a different
% temporal frequency for each spatial frequency according to the optical
% flow equation (ft = v*fx) and applying the spatio-temporal CSF of Kelly.
% 
%
% The program returs:
%  (1) Data of the different 2D-CSFs as a set of concatenated matrices (one per considered speed).
%
%  (2) The values the 2D function in the fx,v plane (a cut of the previous one for fy=0) and the 
%      1d variables fx and v to represent this plane with imagesc.
%
% SYNTAX:  [csfs_at_speeds,csf_fx_v,fx,v] = spatial_csf_speed(fsx,fsy,Nx,Ny,[v_min v_max],Nv,track?,stabiliz?);
%
% NOTE: in order to make explicit the effect at v=0, please select the
% appropriate number of speeds so that v=0 is taken (e.g. off number when the range is symmetric).
%

% [fx,fy,ft]=dominio_freq_espacio_temp(fse,fst,Nx,Nx,Nt);
[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(Ny,Nx,Nv,fsx,fsy,1);

for i=1:Nv
    f=sacafot(ft,Ny,Nx,i);
    ftt(i)=f(1,1);
end

v = linspace(v_range(1),v_range(2),Nv);

F=sqrt(fx.^2+fy.^2);

if track == 0
    
    for i=1:Nv
        f=sacafot(F,Ny,Nx,i);
        ft=metefot(ft,abs(v(i))*f,i,1);
    end
    
    ft=abs(ft)+0.0000000000001;
    F=F+0.0000000000001;
    
    if estab==1  % KELLY'S CSF
        % disp('lala')
        CSFet=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);
    else
        ft=abs(ft)+0.1*F;
        CSFet=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);
    end
    
else
    
    v_ef = ones(size(F));
    for i=1:Nv
        ve = sacafot(v_ef,Ny,Nx,i);
        ve = abs(v(i))*ve-min(0.82*abs(v(i))+0.15,80);     % DALY'S TRACKING FORMULA
        v_ef=metefot(v_ef,ve,i,1);
    end
    
%     c0 = 1.14;   % DALY'S PARAMETERS
%     c1 = 0.67;
%     c2 = 1.7;
%     fm = 45.9./(c2*v_ef+2);
%     
%     CSFet=c0.*c2.*abs(v_ef).*((2*pi*abs(F)*c1).^2).*exp(-(4*pi*c1*abs(F))./fm).*(6.1+7.3*(abs(log10(c2*abs(v_ef)/3))).^3);
    
    for i=1:Nv
        f=sacafot(F,Ny,Nx,i);
        ve=sacafot(v_ef,Ny,Nx,i);
        
        ft=metefot(ft,abs(ve(1,1))*f,i,1);
    end
    
    ft=abs(ft)+0.0000000000001;
    F=F+0.0000000000001;
    
    if estab==1  % KELLY'S CSF with no stabilization (stabilization makes no sense when you allow tracking)
        % disp('lala')
        % CSFet=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);
        
        ft=abs(ft)+0.1*F;
        CSFet=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);
        
    else
        ft=abs(ft)+0.1*F;
        CSFet1=4*pi^2*F.*abs(ft).*exp(-4*pi*(ft+2*F)/45.9).*(6.1+7.3*abs(log10(abs(ft)./(3*F))).^3);

         c0 = 1.14;   % DALY'S PARAMETERS
         c1 = 0.67;
         c2 = 1.7;
         fm = 45.9./(c2*v_ef+2);

         CSFet=c0.*c2.*abs(v_ef).*((2*pi*abs(F)*c1).^2).*exp(-(4*pi*c1*abs(F))./fm).*(6.1+7.3*(abs(log10(c2*abs(v_ef)/3))).^3);

         %A1 = sum(sum(CSFet1));
         %A = sum(sum(CSFet));
         %CSFet = A1*CSFet/A;
     
    end    
    
end

[m,I]=min( abs(fy(:,1) - 0) );

csf_fx_v=zeros(Nx,Nv);
for i=1:Nx
    csf_fx_v(i,:)=slineat(CSFet,[I i]);
end 

fxx=fx(1,1:Nx);