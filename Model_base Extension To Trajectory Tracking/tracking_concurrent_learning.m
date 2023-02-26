%% Model based Tracking Trajectory for Linear System
clc; clear; close all;
%% Time step
Step = 0.0001;
T_end = 10;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
xd = data;
theta = data;
Wa = data;
Wc = data;
u = data;
xm = data;
GAMMA = data;
%% Parameter
g = [0;1];
G = [0;1;0;0];
Q = diag([10 10]);
R = 1;
k = 30;
kc1 = 5;
kc2 = 15;
ka1 = 1;
ka2 = 3;
beta = 0.1;
p_ = 30;
nuy = 0.005;
GAMMA_ = 500;
GAMMA_theta = 30;
k_theta = 20;
%% Initial value
x{1} = [1;1];
xd{1} = [0;2];
theta{1} = randn([2 2]);
Wa{1} = randn([7 1]);
Wc{1} = randn([7 1]);
xm{1} = randn([2 1]);
GAMMA{1} = 100*eye(7);
p = 0;
psi_record = cell(1,p_);
u_record = cell(1,p_);
dX_record = cell(1,p_);
%% Simulation
for i = 1:length(t)
    ud = pinv(g)*(hd_fun(xd{i}) - theta{i}'*sigma_theta(x{i}));
    e = x{i} - xd{i};
    psi = [e;xd{i}];
    muy = -1/2*R^-1*G'*d_sigma(psi)'*Wa{i};
    u{i} = ud + muy;
    x_nga = x{i} - xm{i};
    dx = f_fun(x{i}) + g*u{i};
    dxd = hd_fun(xd{i});
    dxm = theta{i}'*sigma_theta(x{i}) + g*u{i} + k*x_nga;
    F_theta = [theta{i}'*sigma_theta(x{i}) - g*pinv(g)*theta{i}'*sigma_theta([0;0]);0;0];
    F1 = [-hd_fun(xd{i}) + g*pinv(g)*hd_fun(xd{i});hd_fun(xd{i})];
    omega = d_sigma(psi)*(F_theta + F1 + G*muy);
    ro = 1 + nuy*omega'*GAMMA{i}*omega;
    delta_hjb = Wc{i}'*omega + e'*Q*e + muy'*R*muy;
    dWc = -kc1*GAMMA{i}*omega/ro*delta_hjb;
    dWa = -ka1*(Wa{i} - Wc{i}) - ka2*Wa{i} + 1/4/ro*(kc1*d_sigma(psi)*G*R^-1*G'*d_sigma(psi)'*Wa{i}*omega')*Wc{i};
    if p~=0
        for j = 1:p
            e_i = psi_record{j}(1:2);
            xd_i = psi_record{j}(3:4);
            F_theta_i = [theta{i}'*sigma_theta(e_i + xd_i) - g*pinv(g)*theta{i}'*sigma_theta([0;0]);0;0];
            F1_i = [-hd_fun(xd_i) + g*pinv(g)*hd_fun(xd_i);hd_fun(xd_i)];
            muy_i = -1/2*R^-1*G'*d_sigma(psi_record{j})'*Wa{i};
            omega_i = d_sigma(psi_record{j})*(F_theta_i + F1_i + G*muy_i);
            ro_i = 1 + nuy*omega_i'*GAMMA{i}*omega_i;
            delta_ti = Wc{i}'*omega_i + e_i'*Q*e_i + muy_i'*R*muy_i;
            dWc = dWc - kc2/p*GAMMA{i}*omega_i/ro_i*delta_ti;
            dWa = dWa + kc2*1/4/p/ro_i*d_sigma(psi_record{j})*G*R^-1*G'*d_sigma(psi_record{j})'*Wa{i}*omega_i'*Wc{i};
        end
    end
    dGAMMA = (beta*GAMMA{i} - 1/ro^2*kc1*GAMMA{i}*(omega*omega')*GAMMA{i})*(norm(GAMMA{i})<=GAMMA_);
    dtheta = GAMMA_theta*sigma_theta(x{i})*x_nga';
    if p ~= 0
        for j = 1:p
            e_i = psi_record{j}(1:2);
            xd_i = psi_record{j}(3:4);
            dtheta = dtheta + k_theta*GAMMA_theta*sigma_theta(e_i + xd_i)*(dX_record{j} - g*u_record{j} - theta{i}'*sigma_theta(e_i + xd_i))';
        end
    end

    if i == 1
        p = p+1;
        l = p;
        psi_record{p} = psi;
        u_record{p} = u{i};
        dX_record{p} = dx;
    else
        Xk = cell2mat(psi_record);
        if (norm(psi - psi_record{l})^2/norm(psi)>=eps)||(rank([Xk psi]) > rank(Xk))
            if p < p_
                p = p+1;
                l = p;
                psi_record{p} = psi;
                u_record{p} = u{i};
                dX_record{p} = dx;
            else
                T = Xk;
                S_old = min(svd(Xk'));
                S = zeros(p,1);
                for j = 1:p
                    Xk(:,j) = psi;
                    S(j) = min(svd(Xk'));
                    Xk = T;
                end
                [maxS, l] = max(S);
                if maxS > S_old
                    psi_record{l} = psi;
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
    xd{i+1} = xd{i} + Step*dxd;
    %% Update identifier
    xm{i+1} = xm{i} + Step*dxm;
%     theta{i+1} = theta{i} + Step*dtheta;
    theta{i+1} = [-1 -0.5;1 -0.5]';
    %% Update actor critic
    Wa{i+1} = Wa{i} + Step*dWa;
    Wc{i+1} = Wc{i} + Step*dWc;
    GAMMA{i+1} = GAMMA{i} + Step*dGAMMA;
end

function a = f_fun(x)
a = [-1 1;-0.5 -0.5]*x;
end

function a = sigma_theta(x)
a = [x(1);x(2)];
end

function a = d_sigma(psi)
e1 = psi(1);
e2 = psi(2);
xd1 = psi(3);
xd2 = psi(4);
a = [2*e1 0 0 0;
    0 2*e2 0 0;
    e2 e1 0 0;
    xd1 0 e1 0;
    0 xd2 0 e2;
    xd2 0 0 e1;
    0 xd1 e2 0];
end

function a = hd_fun(xd)
a = [-1 1;-2 -1]*xd;
end