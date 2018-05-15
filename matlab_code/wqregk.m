function Obj = wqregk(c)
global N tau Y X T eta_draw

Obj=0;

for t=1:T
    Obj=Obj+mean((Y(:,t)-(c'*[ones(N,1) X(:,:,t) eta_draw]')').*...
        (tau-(Y(:,t)-(c'*[ones(N,1) X(:,:,t) eta_draw]')'<0)));
end
end
