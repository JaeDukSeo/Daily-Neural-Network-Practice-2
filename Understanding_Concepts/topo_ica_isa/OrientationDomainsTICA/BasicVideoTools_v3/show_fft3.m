function show_fft3(tf,fse,fst,fig)

%
% SHOW_FFT3 graphically represents a 3D Fourier transform
%
% The input 3D Fourier transform is in 2D array format (see help of now2then 
% for additional details on the 2D format).
%
% Restrictions:
%  . It assumes squared frames (as fft3 and ifft3)
%  . It requires real (not complex) input, so it requires a decision from
%    the user to choose from abs(F), real(F), etc. where F is the Fourier
%    transform.
%
% SYNTAX: show_fft3( F, fsx, fst, fig)
%
%     F   = 3D Fourier transform data (in 2D array format)
%     fsx = spatial sampling frequency (e.g. in cycl/deg)
%     fst = temporal sampling frequency (e.g. in Hz)
%     fig = figure where the plot will be displayed
%

tf=abs(tf);

m=size(tf);
nf=m(2)/m(1);

%[fx,fy,ft]=dominio_freq_espacio_temp(fse,fst,m(1),m(1),nf);
[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(m(1),m(1),nf,fse,fse,fst);

Fx=zeros(m(1),m(1),nf);
Fy=zeros(m(1),m(1),nf);
Ft=zeros(m(1),m(1),nf);
TF=zeros(m(1),m(1),nf);

for i=1:nf
    fotog=sacafot(fx,m(1),m(1),i);
    Fx(:,:,i)=fotog;
    fotog=sacafot(fy,m(1),m(1),i);
    Fy(:,:,i)=fotog;
    fotog=sacafot(ft,m(1),m(1),i);
    Ft(:,:,i)=fotog;
    fotog=sacafot(tf,m(1),m(1),i);
    TF(:,:,i)=fotog;
end

figure(fig);clf
% contourslice(Fx,Fy,Ft,TF,linspace(min(min(fx)),max(max(fx)),20),[],[])
% xlabel('f_x'),ylabel('f_y'),zlabel('f_t'),
% axis([min(min(fx)) max(max(fx)) min(min(fy)) max(max(fy)) min(min(ft)) max(max(ft))])
% %axis equal
% 
% figure(fig+1);
p = patch(isosurface(Fx, Fy, Ft, TF, max(max(max(TF)))/4));
isonormals(Fx,Fy,Ft,TF, p)
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
daspect([1 1 1])
view(3)
camlight; lighting phong,

% p = patch(isosurface(Fx, Fy, Ft, Ft, 0));
% isonormals(Fx,Fy,Ft,TF, p)
% set(p, 'FaceColor', 'green', 'EdgeColor', 'none');
% alpha(0.5)
% daspect([1 1 1])
% view(3)
% camlight; lighting phong,
% box on,

pp=patch([min(min(fx)) max(max(fx)) max(max(fx)) min(min(fx))],[min(min(fy)) min(min(fy)) max(max(fy)) max(max(fy))],[0 0 0 0],'b');
set(pp,'EdgeColor', 'none','FaceAlpha',0.6);
hold on
plot3([min(min(fx)) max(max(fx))],[0 0],[0 0],'k-')
plot3([0 0],[min(min(fy)) max(max(fy))],[0 0],'k-')
plot3([0 0],[0 0],[min(min(ft)) max(max(ft))],'k-')
box on,
xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),zlabel('f_t (Hz)'),
axis([min(min(fx)) max(max(fx)) min(min(fy)) max(max(fy)) min(min(ft)) max(max(ft))])
hold off
%axis equal
