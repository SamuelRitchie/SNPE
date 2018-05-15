function fval=posterior_hermite(eta_draw)
global N Y X Xbar T Resqinit Resqinit_eta Vectau Ntau  b1 bL b1_eta bL_eta Kx K4

%Likelihood of Y_i
dens=zeros(N,T);
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 X(:,kk,t).*hermite(kk4,eta_draw)];
        end
    end
    for jtau=1:Ntau-1
        dens(:,t)=dens(:,t)+(Vectau(jtau+1)-Vectau(jtau))./((Resqinit(:,jtau+1)-Resqinit(:,jtau))'*X1')'.*...
            (Y(:,t)>(Resqinit(:,jtau)'*X1')').*(Y(:,t)<=(Resqinit(:,jtau+1)'*X1')');
    end

        %Modelling of the first and the last quantile using Laplace
        %distributions        
        
        dens(:,t)= dens(:,t)+Vectau(1)*b1*exp(b1*(Y(:,t)-(Resqinit(:,1)'*X1')')).*(Y(:,t)<=(Resqinit(:,1)'*X1')')+...
            (1-Vectau(Ntau))*bL*exp(-bL*(Y(:,t)-(Resqinit(:,Ntau)'*X1')')).*(Y(:,t)>(Resqinit(:,Ntau)'*X1')');
end

denstot=ones(N,1);
for tt=1:T
    denstot=denstot.*dens(:,tt);
end


%Likelihood of eta_i
dens2=zeros(N,1);
  for jtau=1:Ntau-1 
        dens2=dens2+(Vectau(jtau+1)-Vectau(jtau))./((Resqinit_eta(:,jtau+1)-Resqinit_eta(:,jtau))'*Xbar')'.*...
            (eta_draw>(Resqinit_eta(:,jtau)'*Xbar')').*(eta_draw<=(Resqinit_eta(:,jtau+1)'*Xbar')');
  end

        %Modelling of the first and the last quantile using Laplace
        %distributions        
        
        dens2= dens2+Vectau(1)*b1_eta*exp(b1_eta*(eta_draw-(Resqinit_eta(:,1)'*Xbar')')).*(eta_draw<=(Resqinit_eta(:,1)'*Xbar')')+...
            (1-Vectau(Ntau))*bL_eta*exp(-bL_eta*(eta_draw-(Resqinit_eta(:,Ntau)'*Xbar')')).*(eta_draw>(Resqinit_eta(:,Ntau)'*Xbar')');

fval=dens2.*denstot;

end

   
