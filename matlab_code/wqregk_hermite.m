function Obj = wqregk_hermite(c)
global N tau Y X T eta_draw Kx K4

Obj=0;

for t=1:T
    X1=[];
    for kk=1:Kx
        for kk4=0:K4
            X1=[X1 X(:,kk,t).*hermite(kk4,eta_draw)];
        end
    end
    Obj=Obj+mean((Y(:,t)-(c'*X1')').*...
        (tau-(Y(:,t)-(c'*X1')'<0)));
end
end
