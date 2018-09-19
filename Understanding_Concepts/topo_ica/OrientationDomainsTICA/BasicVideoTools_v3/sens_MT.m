function Gt = sens_MT(tam_fx,tam_fy,tam_ft,fsx,fsy,fst,v)

%
% SENS_MT computes the square of the receptive field of linear MT neurons 
% tuned to certain speed v = [vx vy] by adding the receptive fields of V1
% sensors with sensitivities in the appropriate frequency plane.
%
% SYNTAX:  G = sens_MT(size_fx,size_fy,size_ft,fsx,fsy,fst,v)
%
%  size_fx  = size of the discrete domain in the fx dimension
%  size_fy  = size of the discrete domain in the fy dimension
%  size_ft  = size of the discrete domain in the ft dimension
%  fsx      = spatial sampling frequency (in cpd)
%  fsy      = spatial sampling frequency (in cpd)
%  fst      = temporal sampling frequency (in Hz)
%  v        = preferred speed
%

%[fx,fy,ft]=dominio_freq_espacio_temp(fse,fst,tam_fx,tam_fy,tam_ft);
[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(tam_fx,tam_fy,tam_ft,fsx,fsy,fst);

[fxo,fyo]=freqspace(9,'meshgrid');
fxo=fsx*fxo/4;
fyo=fsy*fyo/4;
Gt=0*fx;

for i=1:size(fxo,1)
    i
    for j=1:size(fxo,2)
        f=[fxo(i,j) fyo(i,j)];
      G =sens_gabor3d(tam_fx,tam_fy,tam_ft,fsx,fsy,fst,f(1),f(2),-v*f',1,1,1);
      Gt=Gt+sqrt(G);

    end
end

G =sens_gabor3d(tam_fx,tam_fy,tam_ft,fsx,fsy,fst,0,0,0,1,1,1);
Gt = Gt - sqrt(G);