clear all
clc;

global N Ntau Y prob X Xbar T Vectau Resqinit Resqinit_eta tau eta_draw b1 bL b1_eta bL_eta


% importing data
data = dlmread('data18.csv');
% load data
%load('data');

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

% covariates.
sumX=zeros(N,Kx);
for t=1:T;
    sumX=sumX+X(:,:,t);
end

% average over time of the covariates for each individual.
Xbar=sumX/T;





% Maximum Iteration
maxiter=100;

% Number of draws within the chain.
draws=200;

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


% Initial conditions

Resqinit=zeros(Kx+2,Ntau);
for jtau=1:Ntau
    tau=Vectau(jtau);
    beta1=rq([ones(N,1) X(:,:,1)],Y(:,1),tau);
    Resqinit(1:Kx+1,jtau)=beta1;
    Resqinit(Kx+2,jtau)=1; %Arbitrary Initial value
end

% Initial conditions for eta process
% Resqinit_eta=kron(ones(Kx+1,1),(.1:.1:.1*Ntau));

% Alternative initial conditions for eta process
Resqinit_eta=zeros(Kx+1,Ntau);
for jtau=1:Ntau
    tau=Vectau(jtau);
    beta1=rq([ones(N,1) Xbar],(Y(:,1)+Y(:,2)+Y(:,3))/3,tau);
    Resqinit_eta(1:Kx+1,jtau)=beta1;
end




% initial conditions: Laplace parameters
%b1=1;
%bL=1;
%b1_eta=1;
%bL_eta=1;

b1=20;
bL=20;
b1_eta=20;
bL_eta=20;



Resqnew=zeros(Kx+2,Ntau,maxiter);
Resqnew_eta=zeros(Kx+1,Ntau,maxiter);


init=randn(N,1);
Obj_chain = [posterior(init) zeros(N,draws-1)];
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
        newObj=posterior(eta_draw);
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
        
        Resqnew_eta(:,jtau,iter)=fminunc(@wqregk_eta,Resqinit_eta(:,jtau),options);
        
        Resqnew(:,jtau,iter)=fminunc(@wqregk,Resqinit(:,jtau),options);
        
    end
    
    %Normalization
    Resqnew(1,:,iter)=Resqnew(1,:,iter)-mean(Resqnew(1,:,iter))-((1-Vectau(Ntau))/bL-Vectau(1)/b1);
    Resqnew(Kx+2,:,iter)=Resqnew(Kx+2,:,iter)-mean(Resqnew(Kx+2,:,iter))+1;
    
    eta_draw_tot=eta_draw;
    for j=2:T
        eta_draw_tot=[eta_draw_tot;eta_draw];
    end
    
    
    % Laplace parameters
    Vect1=Ytot-[ones(N*T*Mdraws,1) Xtot eta_draw_tot]*Resqnew(:,1,iter);
    Vect2=Ytot-[ones(N*T*Mdraws,1) Xtot eta_draw_tot]*Resqnew(:,Ntau,iter);
    b1=-sum(Vect1<=0)/sum(Vect1.*(Vect1<=0));
    bL=sum(Vect2>=0)/sum(Vect2.*(Vect2>=0));
    
    
    Vect1=eta_draw-[ones(N*Mdraws,1) Xbartot]*Resqnew_eta(:,1,iter);
    Vect2=eta_draw-[ones(N*Mdraws,1) Xbartot]*Resqnew_eta(:,Ntau,iter);
    b1_eta=-sum(Vect1<=0)/sum(Vect1.*(Vect1<=0));
    bL_eta=sum(Vect2>=0)/sum(Vect2.*(Vect2>=0));
    
    
    warning on
    
    % Criterion
    
    Resqinit=Resqnew(:,:,iter)
    Resqinit_eta=Resqnew_eta(:,:,iter)
    
    [b1 bL b1_eta bL_eta]
    
    Obj_chain= [Obj_chain(:,draws) zeros(N,draws-1)];
    Nu_chain = [Nu_chain(:,draws) zeros(N,draws-1)];
    acc=zeros(N,draws);
    acceptrate=zeros(draws,1);
end


Resqfinal=zeros(Kx+1,Ntau);
for jtau=1:Ntau
    for p=1:Kx+2
        Resqfinal(p,jtau)=mean(Resqnew(p,jtau,(maxiter/2):maxiter));
    end
end

Resqfinal_eta=zeros(Kx+1,Ntau);
for jtau=1:Ntau
    for p=1:Kx+1
        Resqfinal_eta(p,jtau)=mean(Resqnew_eta(p,jtau,(maxiter/2):maxiter));
    end
end


Resqbasic=zeros(Kx+1,Ntau);
for jtau=1:Ntau
    
    tau=Vectau(jtau);
    
    Resqbasic(:,jtau)=rq([ones(N*T,1) [X(:,1,1);X(:,1,2);X(:,1,3);X(:,1,4);X(:,1,5);X(:,1,6);X(:,1,7);X(:,1,8)]...
        [X(:,2,1);X(:,2,2);X(:,2,3);X(:,2,4);X(:,2,5);X(:,2,6);X(:,2,7);X(:,2,8)] [X(:,3,1);X(:,3,2);X(:,3,3);X(:,3,4);X(:,3,5);X(:,3,6);X(:,3,7);X(:,3,8)]], [Y(:,1);Y(:,2);Y(:,3);Y(:,4);Y(:,5);Y(:,6);Y(:,7);Y(:,8)],tau);
    
    
end

%save all

% QR coefficients
plot(Vectau,Resqfinal(2,:),':k','Linewidth',5)
hold on
plot(Vectau,Resqbasic(2,:),'-k','Linewidth',5)
axis([0 1 -.25 0.25])
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
    Ysim0(:,t)=Resqbasic(1,U1(:,t))'+Resqbasic(2,U1(:,t))'.*X(:,1,t)+...
        Resqbasic(3,U1(:,t))'.*X(:,2,t)+Resqbasic(4,U1(:,t))'.*X(:,3,t);
end

% eta
etasim=Resqfinal_eta(1,V1)'+Resqfinal(2,V1)'.*Xbar(:,1)+...
    Resqfinal(3,V1)'.*Xbar(:,2)+Resqfinal(4,V1)'.*Xbar(:,3);
etasim=etasim+(1/b1_eta)*log(V/Vectau(1)).*(V<Vectau(1))...
    -(1/bL_eta)*log((1-V)/(1-Vectau(Ntau))).*(V>=Vectau(Ntau));

% Y
Ysim=zeros(N,T);
for t=1:T
    Ysim(:,t)=Resqfinal(1,U1(:,t))'+Resqfinal(2,U1(:,t))'.*X(:,1,t)+...
        Resqfinal(3,U1(:,t))'.*X(:,2,t)+Resqfinal(4,U1(:,t))'.*X(:,3,t)+Resqfinal(5,U1(:,t))'.*etasim;
    Ysim(:,t)=Ysim(:,t)+(1/b1)*log(U(:,t)/Vectau(1)).*(U(:,t)<Vectau(1))...
         -(1/bL)*log((1-U(:,t))/(1-Vectau(Ntau))).*(U(:,t)>=Vectau(Ntau));
end

% potential outcomes
YsimN=zeros(N,T);
for t=1:T
    YsimN(:,t)=Resqfinal(1,U1(:,t))'+Resqfinal(2,U1(:,t))'.*min(X(:,1,t))+...
        Resqfinal(3,U1(:,t))'.*X(:,2,t)+Resqfinal(4,U1(:,t))'.*X(:,3,t)+Resqfinal(5,U1(:,t))'.*etasim;
    YsimN(:,t)=YsimN(:,t)+(1/b1)*log(U(:,t)/Vectau(1)).*(U(:,t)<Vectau(1))...
         -(1/bL)*log((1-U(:,t))/(1-Vectau(Ntau))).*(U(:,t)>=Vectau(Ntau));
end

YsimY=zeros(N,T);
for t=1:T
    YsimY(:,t)=Resqfinal(1,U1(:,t))'+Resqfinal(2,U1(:,t))'.*max(X(:,1,t))+...
        Resqfinal(3,U1(:,t))'.*X(:,2,t)+Resqfinal(4,U1(:,t))'.*X(:,3,t)+Resqfinal(5,U1(:,t))'.*etasim;
    YsimY(:,t)=YsimY(:,t)+(1/b1)*log(U(:,t)/Vectau(1)).*(U(:,t)<Vectau(1))...
         -(1/bL)*log((1-U(:,t))/(1-Vectau(Ntau))).*(U(:,t)>=Vectau(Ntau));
end

hold off
plot(Vectau,quantile(YsimN(:),Vectau),'b')
hold on
plot(Vectau,quantile(YsimY(:),Vectau),'r')

VectN=zeros(0,1);
VectY=zeros(0,1);
for t=1:T
    VectN=[VectN;Y(X(:,1,t)==min(X(:,1,t)),t)];
    VectY=[VectY;Y(X(:,1,t)==max(X(:,1,t)),t)];
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

