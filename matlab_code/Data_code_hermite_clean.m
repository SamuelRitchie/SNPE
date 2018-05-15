clear all
clc;

global N Ntau Y prob X Xbar T Vectau Resqinit Resqinit_eta tau eta_draw b1 bL b1_eta bL_eta Kx K4

% importing data
data = dlmread('data18.csv');

Logwage=data(:,3);
Union=data(:,2);
Married=data(:,1);
Expsq=data(:,4);


T = 8;
N = size(Logwage,1)/T;

% Regressors.
Kx=3;

% de-mean covariates
Y=zeros(N,T);
X=zeros(N,Kx,T);
for t=1:T
    Y(:,t)=Logwage(t:T:N*T);
    X(:,1,t)=Union(t:T:N*T);
    X(:,2,t)=Married(t:T:N*T);
    X(:,3,t)=Expsq(t:T:N*T);
    Y(:,t)=Y(:,t)-mean(Y(:,t));
    X(:,1,t)=X(:,1,t)-mean(X(:,1,t));
    X(:,2,t)=X(:,2,t)-mean(X(:,2,t));
    X(:,3,t)=X(:,3,t)-mean(X(:,3,t));
end

Xinit=X;

% covariates.
sumX=zeros(N,Kx);
for t=1:T;
    sumX=sumX+X(:,:,t);
end

% average over time of the covariates for each individual.
Xbar=sumX/T;

% Degree Hermite polynomials
K1=1;
K2=1;
K3=1;
K4=1;

M1=1;
M2=1;
M3=1;

% Create matrices of covariates
XX=[];
for kk1=0:K1
    for kk2=0:K2
        for kk3=0:K3
            XX=[XX hermite(kk1,X(:,1,:)).*hermite(kk2,X(:,2,:)).*...
                hermite(kk3,X(:,3,:))];
        end
    end
end

XXbar=[];
for kk1=0:K1
    for kk2=0:K2
        for kk3=0:K3
            XXbar=[XXbar hermite(kk1,Xbar(:,1)).*hermite(kk2,Xbar(:,2)).*...
                hermite(kk3,Xbar(:,3))];
        end
    end
end

X=XX;
Xbar=XXbar;

Kx=(K1+1)*(K2+1)*(K3+1);

% Maximum Iteration
maxiter=100;

% Number of draws within the chain.
draws=500;

% Number of draws kept for computation.
Mdraws=1;

% define useful vectors
Ytot=kron(ones(Mdraws,1),Y(:,1));
for j=2:T
    Ytot=[Ytot;kron(ones(Mdraws,1),Y(:,j))];
end

Xtot=kron(ones(Mdraws,1),X(:,:,1));
for j=2:T
    Xtot=[Xtot;kron(ones(Mdraws,1),X(:,:,j))];
end

Xbartot=kron(ones(Mdraws,1),Xbar);


% variance Random walk proposals
var_prop=.05;


% Grid of tau's
Ntau=21;
Vectau=(1/(Ntau+1):1/(Ntau+1):Ntau/(Ntau+1))';



% ESTIMATION %

%%%%%%%%%%%%%%% %%%%%%%%%%%%%%% STOCHASTIC EM algorithm %%%%%%%%%%%%%%% %%%%%%%%%%%%%%%

count=1;
deltapar=1;




% Initial conditions (from simple linear model)

Mat=[ -0.1685   -0.1313   -0.1031   -0.0824   -0.0647   -0.0492   -0.0354   -0.0233   -0.0118   -0.0001    0.0110    0.0222  0.0328    0.0436    0.0549    0.0676    0.0809    0.0955    0.1145    0.1394    0.1746;...
  -0.1540   -0.0738   -0.0607   -0.0634   -0.0660   -0.0683   -0.0666   -0.0615   -0.0585   -0.0537   -0.0484   -0.0458 -0.0434   -0.0408   -0.0391   -0.0382   -0.0372   -0.0364   -0.0390   -0.0311   -0.0159;...
  0.0067    0.0054    0.0044    0.0039    0.0035    0.0031    0.0029    0.0027    0.0028    0.0029    0.0028    0.0027   0.0026    0.0025    0.0024    0.0022    0.0022    0.0024    0.0025    0.0024    0.0019;...
    0.0414    0.0362    0.0352    0.0375    0.0396    0.0405    0.0399    0.0403    0.0394    0.0385    0.0373    0.0369  0.0373    0.0373    0.0387    0.0405    0.0409    0.0398    0.0394    0.0381    0.0408;...
    1.2479    1.2004    1.1495    1.1024    1.0637    1.0316    1.0127    1.0059    0.9990    0.9935    0.9883    0.9824  0.9746    0.9605    0.9497    0.9394    0.9249    0.9138    0.8820    0.8601    0.8179];

 
  
   

Resqinit=zeros(Kx*(K4+1),Ntau);
Resqinit(1,:)=Mat(1,:);
Resqinit(2,:)=Mat(5,:);
Resqinit(3,:)=Mat(4,:);
Resqinit(5,:)=Mat(3,:);
Resqinit(9,:)=Mat(2,:);

% Initial conditions for eta process (from simple linear model)

Mateta=[ -0.1645   -0.1233   -0.0948   -0.0735   -0.0571   -0.0446   -0.0341   -0.0243   -0.0156   -0.0074    0.0009    0.0088   0.0179    0.0284    0.0399    0.0521    0.0661    0.0815    0.0993    0.1212    0.1524;...
  -0.0917   -0.0778   -0.0681   -0.0618   -0.0551   -0.0517   -0.0511   -0.0510   -0.0512   -0.0511   -0.0498   -0.0480 -0.0438   -0.0405   -0.0345   -0.0316   -0.0331   -0.0358   -0.0368   -0.0442   -0.0490;...
      0.0004   -0.0002   -0.0000    0.0001    0.0005    0.0006    0.0006    0.0006    0.0005    0.0004    0.0002    0.0001  0.0004    0.0008    0.0012    0.0014    0.0015    0.0015    0.0014    0.0009    0.0014;...
     0.0062    0.0125    0.0016   -0.0061   -0.0075   -0.0034   -0.0027   -0.0018    0.0011    0.0058    0.0098    0.0087 0.0036   -0.0030   -0.0097   -0.0176   -0.0221   -0.0141   -0.0003    0.0003    0.0059]; 

 
  
   
   
    


   
Resqinit_eta=zeros(Kx,Ntau); 

Resqinit_eta(1,:)=Mateta(1,:);
Resqinit_eta(2,:)=Mateta(4,:);
Resqinit_eta(3,:)=Mateta(3,:);
Resqinit_eta(5,:)=Mateta(2,:);
   
  
   


% initial conditions: Laplace parameters
b1=5;
bL=16;
b1_eta=8;
bL_eta=28;

Resqnew=zeros(Kx*(K4+1),Ntau,maxiter);
Resqnew_eta=zeros(Kx,Ntau,maxiter);


init=randn(N,1);
Obj_chain = [posterior_hermite(init) zeros(N,draws-1)];
Nu_chain = ones(N,draws).*((init)*ones(1,draws));
acc=zeros(N,draws);
acceptrate=zeros(draws,1);

for iter=1:maxiter
    iter
    
    %E step
    
    %%%%%%%%%%%%%%% Metropolis-Hastings %%%%%%%%%%%%%%%
    j = 2;
    while j <= draws
        % Proposal and acceptance rule
        eta_draw=Nu_chain(:,j-1)+sqrt(var_prop)*randn(N,1);
        newObj=posterior_hermite(eta_draw);
        r=(min([ones(N,1) newObj./Obj_chain(:,j-1)]'))';
        prob=rand(N,1);
        Obj_chain(:,j)=(prob<=r).*newObj+(prob>r).*Obj_chain(:,j-1);
        Nu_chain(:,j)=(prob<=r).*eta_draw(:,1)+(prob>r).*Nu_chain(:,j-1);
        eta_draw=Nu_chain(:,j);
        acc(:,j)=(prob<=r);
        
        acceptrate(j)=mean(acc(:,j));
        j = j+1;
    end
    
    %Last draws of the chain will be the fixed associated with our data.
    eta_draw=Nu_chain(:,draws-20*(Mdraws-1));
    
    for jj=2:Mdraws
        eta_draw=[eta_draw;Nu_chain(:,draws-20*(Mdraws-jj))];
    end
    
    
    options.Display ='off';
    warning off
    
    for jtau=1:Ntau
        
        tau=Vectau(jtau);
        
        Resqnew_eta(:,jtau,iter)=fminunc(@wqregk_eta_hermite,Resqinit_eta(:,jtau),options);
        
        Resqnew(:,jtau,iter)=fminunc(@wqregk_hermite,Resqinit(:,jtau),options);
        
    end
    
    % Normalization (conditional median=eta)


    Resqnew(1,:,iter)=Resqnew(1,:,iter)-mean(Resqnew(1,:,iter))-((1-Vectau(Ntau))/bL-Vectau(1)/b1);
    Resqnew(2,:,iter)=Resqnew(2,:,iter)-mean(Resqnew(2,:,iter))+1;
    
    
    eta_draw_tot=eta_draw;
    for j=2:T
        eta_draw_tot=[eta_draw_tot;eta_draw];
    end
    
    
    % Laplace parameters
    
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 Xtot(:,kk).*hermite(kk4,eta_draw_tot)];
        end
    end
    
    
    Vect1=Ytot-X1*Resqnew(:,1,iter);
    Vect2=Ytot-X1*Resqnew(:,Ntau,iter);
    b1=-sum(Vect1<=0)/sum(Vect1.*(Vect1<=0));
    bL=sum(Vect2>=0)/sum(Vect2.*(Vect2>=0));
    
    
    Vect1=eta_draw-Xbartot*Resqnew_eta(:,1,iter);
    Vect2=eta_draw-Xbartot*Resqnew_eta(:,Ntau,iter);
    b1_eta=-sum(Vect1<=0)/sum(Vect1.*(Vect1<=0));
    bL_eta=sum(Vect2>=0)/sum(Vect2.*(Vect2>=0));
    
    
    warning on
    
    % Criterion
    
    Resqinit=Resqnew(:,:,iter)
    Resqinit_eta=Resqnew_eta(:,:,iter)
    
    [b1 bL b1_eta bL_eta]
    
    acceptrate
    
    Obj_chain= [Obj_chain(:,draws) zeros(N,draws-1)];
    Nu_chain = [Nu_chain(:,draws) zeros(N,draws-1)];
    acc=zeros(N,draws);
    acceptrate=zeros(draws,1);
    
    
    plot(reshape(Resqnew(2,1,1:iter),iter,1))
    pause(1)
end


Resqfinal=zeros(Kx+1,Ntau);
for jtau=1:Ntau
    for p=1:Kx*(K4+1)
        Resqfinal(p,jtau)=mean(Resqnew(p,jtau,(maxiter/2):maxiter));
    end
end

Resqfinal_eta=zeros(Kx+1,Ntau);
for jtau=1:Ntau
    for p=1:Kx
        Resqfinal_eta(p,jtau)=mean(Resqnew_eta(p,jtau,(maxiter/2):maxiter));
    end
end


Resqbasic=zeros(3+1,Ntau);
for jtau=1:Ntau
    
    tau=Vectau(jtau);
    
    Resqbasic(:,jtau)=rq([ones(N*T,1) [Xinit(:,1,1);Xinit(:,1,2);Xinit(:,1,3);Xinit(:,1,4);Xinit(:,1,5);Xinit(:,1,6);Xinit(:,1,7);Xinit(:,1,8)]...
        [Xinit(:,2,1);Xinit(:,2,2);Xinit(:,2,3);Xinit(:,2,4);Xinit(:,2,5);Xinit(:,2,6);Xinit(:,2,7);Xinit(:,2,8)] [Xinit(:,3,1);Xinit(:,3,2);Xinit(:,3,3);Xinit(:,3,4);Xinit(:,3,5);Xinit(:,3,6);Xinit(:,3,7);Xinit(:,3,8)]], [Y(:,1);Y(:,2);Y(:,3);Y(:,4);Y(:,5);Y(:,6);Y(:,7);Y(:,8)],tau);
    
    
end

%save all

% QR coefficients
plot(Vectau,Resqfinal(2,:),':k','Linewidth',5)
hold on
plot(Vectau,Resqbasic(2,:),'-k','Linewidth',5)
axis([0 1 -.25 0])
xlabel('percentile \tau')
ylabel('union effect')
hold off

% Simulate
eta_draw=Nu_chain(:,1);
U=rand(N,T);
U1=(1+floor(U*(Ntau)));
V=rand(N,1);
V1=(1+floor(V*(Ntau)));

Ysim0=zeros(N,T);
for t=1:T
    Ysim0(:,t)=Resqbasic(1,U1(:,t))'+Resqbasic(2,U1(:,t))'.*Xinit(:,1,t)+...
        Resqbasic(3,U1(:,t))'.*Xinit(:,2,t)+Resqbasic(4,U1(:,t))'.*Xinit(:,3,t);
end

% eta
etasim=Resqfinal_eta(1,V1)'.*Xbar(:,1);
for kk=2:Kx
    etasim=etasim+Resqfinal_eta(kk,V1)'.*Xbar(:,kk);
end
etasim=etasim+(1/b1_eta)*log(V/Vectau(1)).*(V<Vectau(1))...
    -(1/bL_eta)*log((1-V)/(1-Vectau(Ntau))).*(V>=Vectau(Ntau));

% Y
Ysim=zeros(N,T);
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 X(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    Ysim(:,t)=Resqfinal(1,U1(:,t))'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        Ysim(:,t)=Ysim(:,t)+Resqfinal(kk,U1(:,t))'.*X1(:,kk);
    end
    
    Ysim(:,t)=Ysim(:,t)+(1/b1)*log(U(:,t)/Vectau(1)).*(U(:,t)<Vectau(1))...
        -(1/bL)*log((1-U(:,t))/(1-Vectau(Ntau))).*(U(:,t)>=Vectau(Ntau));
end

% potential outcomes

XX=zeros(N,Kx,T);
for t=1:T
    XX1=[];
    for kk1=0:K1
        for kk2=0:K2
            for kk3=0:K3
                XX1=[XX1 hermite(kk1,min(Xinit(:,1,t))).*hermite(kk2,Xinit(:,2,t)).*...
                    hermite(kk3,Xinit(:,3,t))];
            end
        end
    end
    XX(:,:,t)=XX1;
end

YsimN=zeros(N,T);
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimN(:,t)=Resqfinal(1,U1(:,t))'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimN(:,t)=YsimN(:,t)+Resqfinal(kk,U1(:,t))'.*X1(:,kk);
    end
    
    YsimN(:,t)=YsimN(:,t)+(1/b1)*log(U(:,t)/Vectau(1)).*(U(:,t)<Vectau(1))...
        -(1/bL)*log((1-U(:,t))/(1-Vectau(Ntau))).*(U(:,t)>=Vectau(Ntau));
end


YsimNmed=zeros(N,T);
qq=(Ntau+1)/2;
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimNmed(:,t)=Resqfinal(1,qq)'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimNmed(:,t)=YsimNmed(:,t)+Resqfinal(kk,qq)'.*X1(:,kk);
    end
end

YsimNup=zeros(N,T);
qq=16;
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimNup(:,t)=Resqfinal(1,qq)'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimNup(:,t)=YsimNup(:,t)+Resqfinal(kk,qq)'.*X1(:,kk);
    end
end

YsimNdown=zeros(N,T);
qq=6;
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimNdown(:,t)=Resqfinal(1,qq)'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimNdown(:,t)=YsimNdown(:,t)+Resqfinal(kk,qq)'.*X1(:,kk);
    end
end

MatYN=zeros(N*T,Ntau);
for jtau=1:Ntau
    tau=Vectau(jtau);
    YsimN1=zeros(N,T);
    for t=1:T
        X1=[];
        for kk=1:Kx
            for kk4=0:K4
                X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
            end
        end
            YsimN1(:,t)=Resqfinal(1,jtau)'.*X1(:,1);
            for kk=2:Kx*(K4+1)
                YsimN1(:,t)=YsimN1(:,t)+Resqfinal(kk,jtau)'.*X1(:,kk);
            end
    
           
        end
    MatYN(:,jtau)=YsimN1(:);
end


XX=zeros(N,Kx,T);
for t=1:T
    XX1=[];
    for kk1=0:K1
        for kk2=0:K2
            for kk3=0:K3
                XX1=[XX1 hermite(kk1,max(Xinit(:,1,t))).*hermite(kk2,Xinit(:,2,t)).*...
                    hermite(kk3,Xinit(:,3,t))];
            end
        end
    end
    XX(:,:,t)=XX1;
end

YsimY=zeros(N,T);
for t=1:T
     X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
     YsimY(:,t)=Resqfinal(1,U1(:,t))'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimY(:,t)=YsimY(:,t)+Resqfinal(kk,U1(:,t))'.*X1(:,kk);
    end
    
    YsimY(:,t)=YsimY(:,t)+(1/b1)*log(U(:,t)/Vectau(1)).*(U(:,t)<Vectau(1))...
        -(1/bL)*log((1-U(:,t))/(1-Vectau(Ntau))).*(U(:,t)>=Vectau(Ntau));
end

YsimYmed=zeros(N,T);
qq=(Ntau+1)/2;
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimYmed(:,t)=Resqfinal(1,qq)'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimYmed(:,t)=YsimYmed(:,t)+Resqfinal(kk,qq)'.*X1(:,kk);
    end
end

YsimYup=zeros(N,T);
qq=16;
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimYup(:,t)=Resqfinal(1,qq)'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimYup(:,t)=YsimYup(:,t)+Resqfinal(kk,qq)'.*X1(:,kk);
    end
end

YsimYdown=zeros(N,T);
qq=6;
for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
        end
    end
    YsimYdown(:,t)=Resqfinal(1,qq)'.*X1(:,1);
    for kk=2:Kx*(K4+1)
        YsimYdown(:,t)=YsimYdown(:,t)+Resqfinal(kk,qq)'.*X1(:,kk);
    end
end

MatYY=zeros(N*T,Ntau);
for jtau=1:Ntau
    tau=Vectau(jtau);
    YsimY1=zeros(N,T);
    for t=1:T
        X1=[];
        for kk=1:Kx
            for kk4=0:K4
                X1=[X1 XX(:,kk,t).*hermite(kk4,etasim)];
            end
        end
            YsimY1(:,t)=Resqfinal(1,jtau)'.*X1(:,1);
            for kk=2:Kx*(K4+1)
                YsimY1(:,t)=YsimY1(:,t)+Resqfinal(kk,jtau)'.*X1(:,kk);
            end
    
           end
    MatYY(:,jtau)=YsimY1(:);
end

hold off
plot(Vectau,quantile(YsimN(:),Vectau),'b')
hold on
plot(Vectau,quantile(YsimY(:),Vectau),'r')

VectN=zeros(0,1);
VectY=zeros(0,1);
for t=1:T
    VectN=[VectN;Y(Xinit(:,1,t)==min(Xinit(:,1,t)),t)];
    VectY=[VectY;Y(Xinit(:,1,t)==max(Xinit(:,1,t)),t)];
end


hold off
plot(Vectau,quantile(VectN,Vectau),'b')
hold on
plot(Vectau,quantile(VectY,Vectau),'r')
hold off



plot(Vectau,quantile(VectY,Vectau)-quantile(VectN,Vectau),'-k','Linewidth',5)
hold on
plot(Vectau,quantile(YsimY(:),Vectau)-quantile(YsimN(:),Vectau),':k','Linewidth',5)
axis([0 1 -.25 0.4])
xlabel('percentile \tau')
ylabel('quantile treatment effect of being in a union')
hold off

% heterogeneous union effects
plot(Vectau,quantile(YsimYmed(:)-YsimNmed(:),Vectau),'-k','Linewidth',5)
hold on
plot(Vectau,quantile(YsimYup(:)-YsimNup(:),Vectau),'-k','Linewidth',5)
hold on
plot(Vectau,quantile(YsimYdown(:)-YsimNdown(:),Vectau),'-k','Linewidth',5)
hold off

plot(Vectau,quantile(MatYY-MatYN,.05),'-k','Linewidth',1)
hold on
plot(Vectau,quantile(MatYY-MatYN,.25),'-k','Linewidth',1)
hold on
plot(Vectau,quantile(MatYY-MatYN,.5),'-k','Linewidth',2)
hold on
plot(Vectau,quantile(MatYY-MatYN,.75),'-k','Linewidth',1)
hold on
plot(Vectau,quantile(MatYY-MatYN,.95),'-k','Linewidth',1)
axis([0 1 -.25 0.4])
xlabel('percentile \tau')
ylabel('union effect')
hold off




