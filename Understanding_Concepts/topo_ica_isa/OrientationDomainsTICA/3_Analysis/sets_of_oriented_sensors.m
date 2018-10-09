function [class] = sets_of_oriented_sensors(xy_buenas,frec_buenas,N_clases,fig,size_dots_x,xmax,size_dots_f,fmax)

% [class] = sets_of_oriented_sensors(xy_buenas,frec_buenas,N_clases,fig,size_dots_x,xmax,size_dots_f,fmax)

angulos = pi + atan2(frec_buenas(:,2),frec_buenas(:,1));
angulos_n = pi + atan2(-frec_buenas(:,2),-frec_buenas(:,1));


ang_class = linspace(0,2*pi - pi/(N_clases),2*N_clases) - pi;
ang_class*180/pi

ang = 0:2:359;
map_hsv = [[ang';ang']/359 ones(360,1) ones(360,1)];
map_rgb = hsv2rgb(map_hsv);

% Clustering

clear class
for i=1:N_clases
    angulo = ang_class(i);
    d = angulos - angulo;
    indices_p1 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos - (angulo+pi);
    indices_p2 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos - (angulo-pi);
    indices_p3 = find( abs(d) <= pi/(2*N_clases) )';
    indices = unique([indices_p1 indices_p2 indices_p3]);
    d = angulos_n - angulo;
    indices_p4 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos_n - (angulo+pi);
    indices_p5 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos_n - (angulo-pi);
    indices_p6 = find( abs(d) <= pi/(2*N_clases) )';
    indices = unique([indices_p1 indices_p2 indices_p3 indices_p4 indices_p5 indices_p6]);
    class(i).indices = indices;
    class(i).xy = xy_buenas(indices,:);
    class(i).frec = frec_buenas(indices,:);
end
angulo = ang_class(N_clases+1);
    d = angulos - angulo;
    indices_p1 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos - (angulo+pi);
    indices_p2 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos - (angulo-pi);
    indices_p3 = find( abs(d) <= pi/(2*N_clases) )';  
    indices = unique([indices_p1 indices_p2 indices_p3]);
    d = angulos_n - angulo;
    indices_p4 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos_n - (angulo+pi);
    indices_p5 = find( abs(d) <= pi/(2*N_clases) )';
    d = angulos_n - (angulo-pi);
    indices_p6 = find( abs(d) <= pi/(2*N_clases) )';
    indices = unique([indices_p1 indices_p2 indices_p3 indices_p4 indices_p5 indices_p6]);    
class(1).indices = [class(1).indices indices];
class(1).xy = [class(1).xy;xy_buenas(indices,:)];
class(1).frec = [class(1).frec;frec_buenas(indices,:)];

length(angulos)
s = 0;
for i=1:N_clases
    s = s + sum(length(class(i).indices));
end
s

z = zeros(1,s);
k=1;
for i = 1:N_clases
    figure(fig-1+i)
    indices = class(i).indices;
    class(i).map = map_rgb;
    class(i).colores = [];
    for j=1:length(indices)
        z(k)=round(359*(pi+atan2(frec_buenas(indices(j),2),frec_buenas(indices(j),1)))/(2*pi))+1;
        %plot(xy_buenas(indices(j),1),xy_buenas(indices(j),2),'.','markersize',size_dots_x,'color',map_rgb(colores_map_rgb(indices(j)),:)), hold on
        plot(xy_buenas(indices(j),1),xy_buenas(indices(j),2),'.','markersize',size_dots_x,'color',map_rgb(z(k),:)), hold on
        class(i).colores = [class(i).colores z(k)];
        k=k+1;
    end
    axis([-0.05 xmax -0.05 xmax]),axis square,axis ij
    xlabel('x (deg)'),ylabel('y (deg)'),
    %title('Centers of TICA sensors in the retinal space')
    set(gcf,'color',[1 1 1])    
end

for i = 1:N_clases
    figure(fig+N_clases+i)
    indices = class(i).indices;
    map = class(i).map;
    colores = class(i).colores;
    f = class(i).frec;    
    for j=1:length(indices)
        plot(f(j,1),f(j,2),'.','markersize',size_dots_f,'color',map(colores(j),:)), hold on
        plot(-f(j,1),-f(j,2),'.','markersize',size_dots_f,'color',map(colores(j),:)), hold on
    end
    axis([-fmax fmax -fmax fmax]),axis square
    xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),
    %title('Centers of TICA sensors in the retinal space')
    set(gcf,'color',[1 1 1])    
end
