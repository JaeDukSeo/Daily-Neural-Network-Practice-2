
p.epsi = 0.001;
E = linspace(0,100,100000);

    e = 0.4;
    K = (p.epsi^e)/(p.epsi^0.5);
    g = -K*((p.epsi + E).^(-e));
    
figure(1),semilogx(E,g,'ro-')    
    
    e = 0.5;
    K = (p.epsi^e)/(p.epsi^0.5);
    g = -K*((p.epsi + E).^(-e));
    
    hold on,semilogx(E,g,'go-')
    
    e = 0.6;
    K = (p.epsi^e)/(p.epsi^0.5);
    g = -K*((p.epsi + E).^(-e)); 
    
    hold on,semilogx(E,g,'bo-')
    
    K=12;
    a =2;
    g = K*2*(atan(a*E)-pi/2)/pi;
    
    hold on,semilogx(E,g,'ko-')
    axis([2*0.001 100 -12.5 0.5])
    xlabel('E'),ylabel('g(E)'),title('Non-linearities in TICA learning')
    h=legend('Lower exponent','Default exponent','Higher exponent','ArcTangent')
    set(h,'box','off','location','southeast')
    set(gcf,'color',[1 1 1])

    e = 0.4;
    K = (p.epsi^e)/(p.epsi^0.5);
    g = -K*((p.epsi + E).^(-e));
    
figure(2),plot(E,g,'ro-')    
    
    e = 0.5;
    K = (p.epsi^e)/(p.epsi^0.5);
    g = -K*((p.epsi + E).^(-e));
    
    hold on,plot(E,g,'go-')
    
    e = 0.6;
    K = (p.epsi^e)/(p.epsi^0.5);
    g = -K*((p.epsi + E).^(-e)); 
    
    hold on,plot(E,g,'bo-')
    
    K=12;
    a =2;
    g = K*2*(atan(a*E)-pi/2)/pi;
    
    hold on,plot(E,g,'ko-')
    axis([0 10 -12.5 0.5])
    xlabel('E'),ylabel('g(E)'),title('Non-linearities in TICA learning')
    h=legend('Lower exponent','Default exponent','Higher exponent','ArcTangent')
    set(h,'box','off','location','southeast')
    set(gcf,'color',[1 1 1])
    
    