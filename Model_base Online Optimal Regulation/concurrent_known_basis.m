%% Model based Actor Critic with known basis
clc; clear; close all;
%% Time step 
Step = 0.0001;
T_end = 10;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
theta = data;
Wa = data;
Wc = data;
u = data;
GAMMA = data;
%% Parameters
Q = eye(2);
R = 1;
p_ = 30;
eps = 0.01;
eta_c1 = 1;
eta_c2 = 15;
eta_a1 = 100;
eta_a2 = 0.1;
nuy = 0.005;
kx = 10*eye(2);
GAMMA_theta = 20*eye(4);
k_theta = 30;
beta = 0.1;
GAMMA_ = 500;
%% Initial value
x{1} = [-1;-1];
theta{1} = randn([4 1]);
X_record = cell(1,p_);
u_record = cell(1,p_);
dX_record = cell(1,p_);
Wa{1} = randn([3 1]) + [1;1;1];
Wc{1} = randn([3 1]) + [1;1;1];
p = 0;
GAMMA{1} = 100*eye(3);
%% Simulation
for i = 1:length(t)
    u{i} = -1/2*R^-1*g_fun(x{i})'*d_sigma(x{i})'*Wa{i};
    dx = f_fun(x{i}) + g_fun(x{i})*u{i};
    delta_hjb = Wc{i}'*d_sigma(x{i})*(Y_fun(x{i})*theta{i} + g_fun(x{i})*u{i}) + x{i}'*Q*x{i} + u{i}'*R*u{i};
    omega = d_sigma(x{i})*(Y_fun(x{i})*theta{i} + g_fun(x{i})*u{i});
    ro = 1 + nuy*omega'*GAMMA{i}*omega;
    dWc = -eta_c1*GAMMA{i}*omega/ro*delta_hjb;
    dWa = -eta_a1*(Wa{i} - Wc{i}) - eta_a2*Wa{i} + 1/4/ro*(eta_c1*d_sigma(x{i})*g_fun(x{i})*R^-1*g_fun(x{i})'*d_sigma(x{i})'*Wa{i}*omega'*Wc{i});
    if p ~= 0
        for j = 1:p
            u_i = -1/2*R^-1*g_fun(X_record{j})'*d_sigma(X_record{j})'*Wa{i};
            omega_i = d_sigma(X_record{j})*(Y_fun(X_record{j})*theta{i} + g_fun(X_record{j})*u_i);
            ro_i = 1+omega_i'*GAMMA{i}*omega_i;
            delta_ti = Wc{i}'*omega_i + (X_record{j}'*Q*X_record{j} + u_i'*R*u_i);
            dWc = dWc - eta_c2/p*GAMMA{i}*omega_i/ro_i*delta_ti;
            dWa = dWa + 1/4/ro_i/p*(eta_c2*d_sigma(X_record{j})*g_fun(X_record{j})*R^-1*g_fun(X_record{j})'*d_sigma(X_record{j})'*Wa{i}*omega_i'*Wc{i});
        end
    end
    dGAMMA = (beta*GAMMA{i} - eta_c1*GAMMA{i}*(omega*omega')*GAMMA{i}/ro^2)*(norm(GAMMA{i})<=GAMMA_);
    dtheta = [0;0;0;0];
    if p ~= 0
        for j = 1:p
            dtheta = dtheta + GAMMA_theta*k_theta/p*Y_fun(X_record{j})'*(dX_record{j} - g_fun(X_record{j})*u_record{j} - Y_fun(X_record{j})*theta{i});
        end
    end
    if i == 1
        p = p+1;
        l = p;
        X_record{p} = x{i};
        u_record{p} = u{i};
        dX_record{p} = dx;
    else
        Xk = cell2mat(X_record);
        if (norm(x{i} - X_record{l})^2/norm(x{i})>=eps)||(rank([Xk x{i}]) > rank(Xk))
            if p < p_
                p = p+1;
                l = p;
                X_record{p} = x{i};
                u_record{p} = u{i};
                dX_record{p} = dx;
            else
                T = Xk;
                S_old = min(svd(Xk'));
                S = zeros(p,1);
                for j = 1:p
                    Xk(:,j) = x{i};
                    S(j) = min(svd(Xk'));
                    Xk = T;
                end
                [maxS, l] = max(S);
                if maxS > S_old
                    X_record{l} = x{i};
                    u_record{l} = u{i};
                    dX_record{l} = dx;
                end
            end
        end
    end
    
    if i == length(t)
        break
    end
    
    %% Update state
    x{i+1} = x{i} + Step*dx;
    %% Update theta
    theta{i+1} = theta{i} + Step*dtheta;
    %% Update actor - critic
    Wa{i+1} = Wa{i} + Step*dWa;
    Wc{i+1} = Wc{i} + Step*dWc;
    GAMMA{i+1} = GAMMA{i} + Step*dGAMMA;
end

x_plot = cell2mat(x);
Wc_plot = cell2mat(Wc);
Wa_plot = cell2mat(Wa);

figure(1);
plot(t,x_plot);
figure(2);
plot(t,Wc_plot);
figure(3);
plot(t,Wa_plot);

function a = f_fun(x)
a = -1;
b = 1;
c = -0.5;
d = -0.5;
x1 = x(1);
x2 = x(2);
a = [x1 x2 0 0;
    0 0 x1 x2*(1-(cos(2*x1)+2)^2)]*[a;b;c;d];
end

function a = g_fun(x)
x1 = x(1);
a = [0;cos(2*x1)+2];
end

function a = d_sigma(x)
x1 = x(1);
x2 = x(2);
a = [2*x1 0;
    x2 x1;
    0 2*x2];
end

function a = Y_fun(x)
x1 = x(1);
x2 = x(2);
a = [x1 x2 0 0;
    0 0 x1 x2*(1-(cos(2*x1)+2)^2)];
end