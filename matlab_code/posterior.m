function fval=posterior(eta_draw)
global N Y X Xbar T Resqinit Resqinit_eta Vectau Ntau  b1 bL b1_eta bL_eta

%Likelihood of Y_i
dens=zeros(N,T);
for t=1:T
  for jtau=1:Ntau-1 
        dens(:,t)=dens(:,t)+(Vectau(jtau+1)-Vectau(jtau))./((Resqinit(:,jtau+1)-Resqinit(:,jtau))'*[ones(N,1) X(:,:,t) eta_draw]')'.*...
            (Y(:,t)>(Resqinit(:,jtau)'*[ones(N,1) X(:,:,t) eta_draw]')').*(Y(:,t)<=(Resqinit(:,jtau+1)'*[ones(N,1) X(:,:,t) eta_draw]')');
  end

        %Modelling of the first and the last quantile using Laplace
        %distributions        
        
        dens(:,t)= dens(:,t)+Vectau(1)*b1*exp(b1*(Y(:,t)-(Resqinit(:,1)'*[ones(N,1) X(:,:,t) eta_draw]')')).*(Y(:,t)<=(Resqinit(:,1)'*[ones(N,1) X(:,:,t) eta_draw]')')+...
            (1-Vectau(Ntau))*bL*exp(-bL*(Y(:,t)-(Resqinit(:,Ntau)'*[ones(N,1) X(:,:,t) eta_draw]')')).*(Y(:,t)>(Resqinit(:,Ntau)'*[ones(N,1) X(:,:,t) eta_draw]')');
end

denstot=ones(N,1);
for tt=1:T
    denstot=denstot.*dens(:,tt);
end


%Likelihood of eta_i
dens2=zeros(N,1);
  for jtau=1:Ntau-1 
        dens2=dens2+(Vectau(jtau+1)-Vectau(jtau))./((Resqinit_eta(:,jtau+1)-Resqinit_eta(:,jtau))'*[ones(N,1) Xbar]')'.*...
            (eta_draw>(Resqinit_eta(:,jtau)'*[ones(N,1) Xbar]')').*(eta_draw<=(Resqinit_eta(:,jtau+1)'*[ones(N,1) Xbar]')');
  end

        %Modelling of the first and the last quantile using Laplace
        %distributions        
        
        dens2= dens2+Vectau(1)*b1_eta*exp(b1_eta*(eta_draw-(Resqinit_eta(:,1)'*[ones(N,1) Xbar]')')).*(eta_draw<=(Resqinit_eta(:,1)'*[ones(N,1) Xbar]')')+...
            (1-Vectau(Ntau))*bL_eta*exp(-bL_eta*(eta_draw-(Resqinit_eta(:,Ntau)'*[ones(N,1) Xbar]')')).*(eta_draw>(Resqinit_eta(:,Ntau)'*[ones(N,1) Xbar]')');

fval=dens2.*denstot;

end

   
