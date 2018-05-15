function Obj = wqregk_eta_hermite(c)
global tau eta_draw Xbar

Obj=mean((eta_draw-(c'*Xbar')').*...
            (tau-(eta_draw-(c'*Xbar')'<0)));
    
end