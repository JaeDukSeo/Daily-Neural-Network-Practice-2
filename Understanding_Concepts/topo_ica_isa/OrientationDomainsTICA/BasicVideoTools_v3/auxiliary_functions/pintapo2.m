function [M]=pintapo2(mm,x,y,pol,col,li)

% PINTAPO2 pinta de color C los pixels de M situados dentro del trapecio
% definido por los puntos 2D [P1;P2;P3;P4] suponiendo que los pixels de M
% muestrean el dominio x,y.
%
% USO: M=pintapo2(M,x,y,[P1;P2;P3;P4],C,lim);

l=size(pol);
l=l(1);
if l==3
   pol=[pol;pol(2,:)];
   a=pol(2,:)-pol(1,:);
   ma=sqrt(a*a');
   b=pol(3,:)-pol(2,:);
   mb=sqrt(b*b');
   c=pol(1,:)-pol(3,:);
   mc=sqrt(c*c');
   d=-a;
else
   a=pol(2,:)-pol(1,:);
   ma=sqrt(a*a');
   b=pol(3,:)-pol(2,:);
   mb=sqrt(b*b');
   c=pol(4,:)-pol(3,:);
   mc=sqrt(c*c');
   d=pol(1,:)-pol(4,:);
   md=sqrt(d*d');
end


if (any((d==-a)==0))     % ES DISTINTO?
   ss=size(mm);
if ((abs(a*b'/(ma*mb))>=li)&(abs(c*b'/(mc*mb))>=li))|((abs(b*c'/(mb*mc))>=li)&(abs(c*d'/(mc*md))>=li))|((abs(c*d'/(mc*md))>=li)&(abs(a*d'/(ma*md))>=li))|((abs(a*d'/(ma*md))>=li)&(abs(a*b'/(ma*mb))>=li))
   M=mm;
else
   pa=pol(1,:);pb=pol(2,:);pc=pol(3,:);pd=pol(4,:);aa=a;
       if aa(1)~=0                         
           m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
           if (pc(2)>(m*pc(1)+n))&(pd(2)>(m*pd(1)+n))
              c1=(y>m*x+n);
           elseif (pc(2)<(m*pc(1)+n))&(pd(2)<(m*pd(1)+n))
              c1=(y<m*x+n);
           else  
              c1=ones(ss(1),ss(2));
           end
       else
           if (pc(1)>pb(1))&(pd(1)>pb(1))
              c1=(x>pb(1));
           elseif (pc(1)<pb(1))&(pd(1)<pb(1))
              c1=(x<pb(1));
           else  
              c1=ones(ss(1),ss(2));
           end       
       end 
  pa=pol(2,:);pb=pol(3,:);pc=pol(4,:);pd=pol(1,:);aa=b;
       if aa(1)~=0                         
           m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
           if (pc(2)>(m*pc(1)+n))&(pd(2)>(m*pd(1)+n))
              c2=(y>m*x+n);
           elseif (pc(2)<(m*pc(1)+n))&(pd(2)<(m*pd(1)+n))
              c2=(y<m*x+n);
           else  
              c2=ones(ss(1),ss(2));
           end
       else
           if (pc(1)>pb(1))&(pd(1)>pb(1))
              c2=(x>pb(1));
           elseif (pc(1)<pb(1))&(pd(1)<pb(1))
              c2=(x<pb(1));
           else  
              c2=ones(ss(1),ss(2));
           end       
       end
  pa=pol(3,:);pb=pol(4,:);pc=pol(1,:);pd=pol(2,:);aa=c;
       if aa(1)~=0                         
           m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
           if (pc(2)>(m*pc(1)+n))&(pd(2)>(m*pd(1)+n))
              c3=(y>m*x+n);
           elseif (pc(2)<(m*pc(1)+n))&(pd(2)<(m*pd(1)+n))
              c3=(y<m*x+n);
           else  
              c3=ones(ss(1),ss(2));
           end
       else
           if (pc(1)>pb(1))&(pd(1)>pb(1))
              c3=(x>pb(1));
           elseif (pc(1)<pb(1))&(pd(1)<pb(1))
              c3=(x<pb(1));
           else  
              c3=ones(ss(1),ss(2));
           end       
       end
  pa=pol(4,:);pb=pol(1,:);pc=pol(2,:);pd=pol(3,:);aa=d;
       if aa(1)~=0                         
           m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
           if (pc(2)>(m*pc(1)+n))&(pd(2)>(m*pd(1)+n))
              c4=(y>m*x+n);
           elseif (pc(2)<(m*pc(1)+n))&(pd(2)<(m*pd(1)+n))
              c4=(y<m*x+n);
           else  
              c4=ones(ss(1),ss(2));
           end
       else
           if (pc(1)>pb(1))&(pd(1)>pb(1))
              c4=(x>pb(1));
           elseif (pc(1)<pb(1))&(pd(1)<pb(1))
              c4=(x<pb(1));
           else  
              c4=ones(ss(1),ss(2));
           end       
       end
  cc=c1.*c2.*c3.*c4;
  M=mm.*(cc<1)+cc*col; 
end
else
   c=pol(1,:)-pol(3,:);
   mc=sqrt(c*c');
   ss=size(mm);
   if ((abs(a*b'/(ma*mb))>=li)&(abs(c*b'/(mc*mb))>=li))|((abs(b*c'/(mb*mc))>=li)&(abs(c*d'/(mc*md))>=li))|((abs(c*d'/(mc*md))>=li)&(abs(a*d'/(ma*md))>=li))|((abs(a*d'/(ma*md))>=li)&(abs(a*b'/(ma*mb))>=li))
      M=mm;
   else
      pa=pol(1,:);pb=pol(2,:);pc=pol(3,:);aa=a;
          if aa(1)~=0                         
              m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
              if pc(2)>(m*pc(1)+n)
                 c1=(y>m*x+n);
              elseif pc(2)<(m*pc(1)+n)
                 c1=(y<m*x+n);
              else  
                 c1=ones(ss(1),ss(2));
              end
          else
              if pc(1)>pb(1)
                 c1=(x>pb(1));
              elseif pc(1)<pb(1)
                 c1=(x<pb(1));
              else  
                 c1=ones(ss(1),ss(2));
              end       
          end 
      pa=pol(2,:);pb=pol(3,:);pc=pol(1,:);aa=b;
          if aa(1)~=0                         
              m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
              if pc(2)>(m*pc(1)+n)
                 c2=(y>m*x+n);
              elseif pc(2)<(m*pc(1)+n)
                 c2=(y<m*x+n);
              else  
                 c2=ones(ss(1),ss(2));
              end
          else
              if pc(1)>pb(1)
                 c2=(x>pb(1));
              elseif pc(1)<pb(1)
                 c2=(x<pb(1));
              else  
                 c2=ones(ss(1),ss(2));
              end       
          end
      pa=pol(3,:);pb=pol(1,:);pc=pol(2,:);aa=c;
          if aa(1)~=0                         
              m=aa(2)/aa(1);n=pb(2)-pb(1)*m;
              if pc(2)>(m*pc(1)+n)
                 c3=(y>m*x+n);
              elseif pc(2)<(m*pc(1)+n)
                 c3=(y<m*x+n);
              else  
                 c3=ones(ss(1),ss(2));
              end
          else
              if pc(1)>pb(1)
                 c3=(x>pb(1));
              elseif pc(1)<pb(1)
                 c3=(x<pb(1));
              else  
                 c3=ones(ss(1),ss(2));
              end       
          end
      cc=c1.*c2.*c3;
      M=mm.*(cc<1)+cc*col;
   end
end