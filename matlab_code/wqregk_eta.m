function Obj = wqregk_eta(c)
global N tau eta_draw Xbar

Obj=mean((eta_draw-(c'*[ones(N,1) Xbar]')').*...
            (tau-(eta_draw-(c'*[ones(N,1) Xbar]')'<0)));
    
end