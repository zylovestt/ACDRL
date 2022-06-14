F=lambda f:(lambda x:si.quad(f,0,x)[0])

def mysgd(fx,x,u,lr=1e-2,error_max=1e-2):
    Fx=F(fx)
    dx=lambda x:(Fx(x)-u)*fx(x)
    diff=dx(x)
    while abs(diff)>error_max:
        x-=lr*diff
        diff=dx(x)
    return x

def mynewton_1(fx,dfx,x,u,error_max=1e-2):
    Fx=F(fx)
    dx=lambda x:(Fx(x)-u)*fx(x)
    ddx=lambda x:(Fx(x)-u)*dfx(x)+fx(x)**2
    newton=lambda x:-1/ddx(x)*dx(x)
    while abs(dx(x))>error_max:
        step=newton(x)
        x+=step
    return x

def mynewton_2(fx,x,u,error_max=1e-2):
    target=lambda x:F(fx)(x)-u
    eps=1e-4
    stepx=lambda x:-target(x)/(fx(x)+eps)
    s=stepx(x)
    while abs(s)>error_max:
        x+=s
        s=stepx(x)
    return x