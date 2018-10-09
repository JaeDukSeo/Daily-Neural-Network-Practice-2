function  [x,y,t,ffx,ffy,ff_t] = spatio_temp_freq_domain(Ny,Nx,Nt,fsx,fsy,fst);

%
% SPATIO_TEMP_FREQ_DOMAIN generates discrete spatio-temporal and 3d-Fourier domains
% of certain extent with certain spatial and temporal sampling frequencies
% These domains allow to generate synthetic sequences and filters in the 3d
% Fourier domain. The domain is arranged in the 2d format (see help of now2then or then2now)
%
%   [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(num_rows,num_cols,num_frames,fsx,fsy,fst);
%
%    num_rows = number of rows in the discrete domain
%    num_cols = number of columns in the discrete domain
%    num_frames = number of frames in the discrete domain
%           The output variables are 2D matrices of size num_rows*(num_cols*num_frames)
%    fsx = spatial sampling frequency in the x -columns- direction (in cycl/deg)
%    fsy = spatial sampling frequency in the y -rows- direction (in cycl/deg)
%    fst = temporal sampling frequency (in Hz)
%
%    

int_x=Nx/fsx;
int_y=Ny/fsy;
int_t=Nt/fst;

x=zeros(Ny,Nx*Nt);
y=zeros(Ny,Nx*Nt);
t=zeros(Ny,Nx*Nt);

fot_x=linspace(0,int_x,Nx+1);
fot_x=fot_x(1:end-1);
fot_x=repmat(fot_x,Ny,1);

fot_y=linspace(0,int_y,Ny+1);
fot_y=fot_y(1:end-1);
fot_y=repmat(fot_y',1,Nx);

fot_t=ones(Ny,Nx);

val_t=linspace(0,int_t,Nt+1);
val_t=val_t(1:end-1);

for i=1:Nt
    x=metefot(x,fot_x,i,1);
    y=metefot(y,fot_y,i,1);    
    t=metefot(t,val_t(i)*fot_t,i,1);    
end    

[fx,fy]=freqspace([Ny Nx],'meshgrid');

fx=fx*fsx/2;
fy=fy*fsy/2;

ffx=zeros(Ny,Nx*Nt);
ffy=zeros(Ny,Nx*Nt);
ff_t=zeros(Ny,Nx*Nt);

fot_fx=fx;
fot_fy=fy;

fot_t=ones(Ny,Nx);

[ft,ft2]=freqspace(Nt);
val_t=ft*fst/2;

for i=1:Nt
    ffx=metefot(ffx,fot_fx,i,1);
    ffy=metefot(ffy,fot_fy,i,1);    
    ff_t=metefot(ff_t,val_t(i)*fot_t,i,1);    
end