function [ima,ESTT,ACET,TTTT,ESTR,ACER,RRR,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,se_ve,p2dx,p2dy]=cargapel(tamx,tamy,tray,fotog)

% CARGAPEL sirve para cargar las peliculitas generadas con PELIROTT.
% Almacena los N fotogramas (de tamaño m*n) uno junto a otro en una
% matriz de tamaño m*(N*n). Tambien carga TODOS los datos de la pelicula.
%
% USO: [PEL,ESTT,ACET,TTTT,ESTR,ACER,RRR,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,se_ve,p2dx,p2dy]=
%                   cargapel(tamaño x,tamaño y,'trayectoria y nombre',[fotog_inic fotog_fin]);
%
% Lo mejor es utilizar PELI2FOT inmediatamente despues de CARGAPEL como por ejemplo asi:
%
% [PEL,ESTT,ACET,TTTT,ESTR,ACER,RRR,ptos0X,ptos0Y,ptos0Z,p3dx,p3dy,p3dz,se_ve,p2dx,p2dy]=
% cargapel(nºfilas,nºcolumnas,'trayectoria y nombre',[fotog_inic fotog_fin]);
% NFOT=[fotog_inic fotog_fin];tamy=nºcolumnas;peli2fot

for i=fotog(1):fotog(2)
      k=i-fotog(1)+1;
      fich=[num2str(i),'.fot'];
      pun=fopen([tray,fich],'r');
      fseek(pun,0,-1);
      [im,c]=fread(pun,[tamx tamy],'uchar');
      fclose(pun);
      ima(:,(k-1)*tamy+1:k*tamy)=im;
end
eval(['load ',tray,'.dat -mat'])