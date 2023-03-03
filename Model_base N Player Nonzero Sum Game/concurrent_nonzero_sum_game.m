%% Model based 2 player nonzero sum game
clc; clear; close all;
%% Time step
Step = 0.0001;
T_end = 10;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
xm = data;
u1 = data;
u2 = data;
Wc1 = data;
GAMMA1 = data;
Wa1 = data;
Wc2 = data;
GAMMA2 = data;
Wa2 = data;
theta = data;
u1_chuan = data;
u2_chuan = data;
%% Parameter
kx = 5;
GAMMA_theta = diag([20 20 20 20 20 20]);
k_theta = 1.5;
p_ = 30;
kc11 = 1;
kc12 = 1;
kc21 = 1;
kc22 = 1;
ka11 = 10;
ka12 = 10;
ka21 = 0;
ka22 = 0;
R11 = 2;
R12 = 2;
R21 = 1;
R22 = 2;
Q1 = 2*eye(2);
Q2 = eye(2);
nuy1 = 0.005;
nuy2 = 0.005;
beta1 = 0.1;
beta2 = 0.1;
GAMMA1_upper = 10000;
GAMMA2_upper = 10000;
eps = 0.005;
%% Initial value
x{1} = [1;1];
xm{1} = randn([2 1]);
theta{1} = 0.5*ones([6 1]) + randn([6 1]);
GAMMA1{1} = 100*eye(3);
GAMMA2{1} = 100*eye(3);
Wc1{1} = [3;3;3] + randn([3 1]);
Wa1{1} = [3;3;3] + randn([3 1]);
Wc2{1} = [3;3;3] + randn([3 1]);
Wa2{1} = [3;3;3] + randn([3 1]);
p = 0;
X_record = cell(1,p_);
U1_record = cell(1,p_);
U2_record = cell(1,p_);
dX_record = cell(1,p_);
%% Simulation4
for i = 1:length(t)
    u1{i} = -1/2*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i};
    u2{i} = -1/2*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i};
%     u1{i} = 0;
%     u2{i} = 0;
    x_nga = x{i} - xm{i};
    dx =  f_fun(x{i}) + g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i};
    dxm = Y_fun(x{i})*theta{i} + g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i} + kx*x_nga;
    dtheta = GAMMA_theta*Y_fun(x{i})'*x_nga;
    if p~=0
        for j = 1:p
            Y_j = Y_fun(X_record{j});
            g1_j = g1_fun(X_record{j});
            g2_j = g2_fun(X_record{j});
            dtheta = dtheta + GAMMA_theta*k_theta*Y_j'*(dX_record{j} - g1_j*U1_record{j} - g2_j*U2_record{j} - Y_j*theta{i});
        end
    end
    
    omega1 = d_sigma(x{i})*Y_fun(x{i})*theta{i} - 1/2*d_sigma(x{i})*g1_fun(x{i})*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i} - 1/2*d_sigma(x{i})*g2_fun(x{i})*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i};
    omega2 = d_sigma(x{i})*Y_fun(x{i})*theta{i} - 1/2*d_sigma(x{i})*g1_fun(x{i})*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i} - 1/2*d_sigma(x{i})*g2_fun(x{i})*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i};
    ro1 = 1 + nuy1*omega1'*GAMMA1{i}*omega1;
    ro2 = 1 + nuy2*omega2'*GAMMA2{i}*omega2;
    delta_hjb1 = Wc1{i}'*d_sigma(x{i})*(Y_fun(x{i})*theta{i} + g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i}) + x{i}'*Q1*x{i} + u1{i}'*R11*u1{i} + u2{i}'*R12*u2{i};
    delta_hjb2 = Wc2{i}'*d_sigma(x{i})*(Y_fun(x{i})*theta{i} + g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i}) + x{i}'*Q2*x{i} + u1{i}'*R21*u1{i} + u2{i}'*R22*u2{i};

    dWc1 = -kc11*GAMMA1{i}*omega1/ro1*delta_hjb1;
    dWc2 = -kc12*GAMMA2{i}*omega2/ro2*delta_hjb2;
    dGAMMA1 = (beta1*GAMMA1{i} - kc11*GAMMA1{i}*(omega1*omega1')/ro1^2*GAMMA1{i})*(norm(GAMMA1{i})<=GAMMA1_upper);
    dGAMMA2 = (beta2*GAMMA2{i} - kc12*GAMMA2{i}*(omega2*omega2')/ro2^2*GAMMA1{i})*(norm(GAMMA2{i})<=GAMMA2_upper);
    dWa1 = -ka11*(Wa1{i} - Wc1{i}) - ka21*Wa1{i} + 1/4*kc11*d_sigma(x{i})*g1_fun(x{i})*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i}*omega1'/ro1*Wc1{i} + 1/4*kc11*d_sigma(x{i})*g2_fun(x{i})*R22^-1*R12*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i}*omega1'/ro1*Wc1{i};
    dWa2 = -ka12*(Wa2{i} - Wc2{i}) - ka22*Wa2{i} + 1/4*kc12*d_sigma(x{i})*g2_fun(x{i})*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i}*omega2'/ro2*Wc2{i} + 1/4*kc12*d_sigma(x{i})*g1_fun(x{i})*R11^-1*R21*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i}*omega2'/ro2*Wc2{i};
    
    if p~=0
        for j = 1:p
            omega1_k = d_sigma(X_record{j})*Y_fun(X_record{j})*theta{i} - 1/2*d_sigma(X_record{j})*g1_fun(X_record{j})*R11^-1*g1_fun(X_record{j})'*d_sigma(X_record{j})'*Wa1{i} - 1/2*d_sigma(X_record{j})*g2_fun(X_record{j})*R22^-1*g2_fun(X_record{j})'*d_sigma(X_record{j})'*Wa2{i};
            omega2_k = d_sigma(X_record{j})*Y_fun(X_record{j})*theta{i} - 1/2*d_sigma(X_record{j})*g1_fun(X_record{j})*R11^-1*g1_fun(X_record{j})'*d_sigma(X_record{j})'*Wa1{i} - 1/2*d_sigma(X_record{j})*g2_fun(X_record{j})*R22^-1*g2_fun(X_record{j})'*d_sigma(X_record{j})'*Wa2{i};
            ro1_k = 1 + nuy1*omega1_k'*GAMMA1{i}*omega1_k;
            ro2_k = 1 + nuy2*omega2_k'*GAMMA2{i}*omega2_k;
            u1_k = -1/2*R11^-1*g1_fun(X_record{j})'*d_sigma(X_record{j})'*Wa1{i};
            u2_k = -1/2*R22^-1*g2_fun(X_record{j})'*d_sigma(X_record{j})'*Wa2{i};
            delta1_k = Wc1{i}'*d_sigma(X_record{j})*(Y_fun(X_record{j})*theta{i} + g1_fun(X_record{j})*u1_k + g2_fun(X_record{j})*u2_k) + X_record{j}'*Q1*X_record{j} + u1_k'*R11*u1_k + u2_k'*R12*u2_k;
            delta2_k = Wc2{i}'*d_sigma(X_record{j})*(Y_fun(X_record{j})*theta{i} + g1_fun(X_record{j})*u1_k + g2_fun(X_record{j})*u2_k) + X_record{j}'*Q2*X_record{j} + u1_k'*R21*u1_k + u2_k'*R22*u2_k;
            dWc1 = dWc1 - kc21*GAMMA1{i}/p*omega1_k/ro1_k*delta1_k;
            dWc2 = dWc2 - kc22*GAMMA2{i}/p*omega2_k/ro2_k*delta2_k;
            dWa1 = dWa1 + 1/4/p*kc21*d_sigma(X_record{j})*g1_fun(X_record{j})*R11^-1*g1_fun(X_record{j})'*d_sigma(X_record{j})'*Wa1{i}*omega1_k'/ro1_k*Wc1{i} + 1/4/p*kc21*d_sigma(X_record{j})*g2_fun(X_record{j})*R22^-1*R12*R22^-1*g2_fun(X_record{j})'*d_sigma(X_record{j})'*Wa2{i}*omega1_k'/ro1_k*Wc1{i};
            dWa2 = dWa2 + 1/4/p*kc22*d_sigma(X_record{j})*g2_fun(X_record{j})*R22^-1*g2_fun(X_record{j})'*d_sigma(X_record{j})'*Wa2{i}*omega2_k'/ro2_k*Wc2{i} + 1/4/p*kc22*d_sigma(X_record{j})*g1_fun(X_record{j})*R11^-1*R21*R11^-1*g1_fun(X_record{j})'*d_sigma(X_record{j})'*Wa1{i}*omega2_k'/ro2_k*Wc2{i};
        end
    end


    if i == 1
        p = p+1;
        l = p;
        X_record{p} = x{i};
        U1_record{p} = u1{i};
        U2_record{p} = u2{i};
        dX_record{p} = dx;
    else
        Xk = cell2mat(X_record);
        if (norm(x{i} - X_record{l})^2/norm(x{i})>=eps)||(rank([Xk x{i}]) > rank(Xk))
            if p < p_
                p = p+1;
                l = p;
                X_record{p} = x{i};
                U1_record{p} = u1{i};
                U2_record{p} = u2{i};
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
                    U1_record{l} = u1{i};
                    U2_record{l} = u2{i};
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
    %% Update identifier
    xm{i+1} = xm{i} + Step*dxm;
    theta{i+1} = theta{i} + Step*dtheta;

    %% Update actor - critic
    Wc1{i+1} = Wc1{i} + Step*dWc1;
    Wc2{i+1} = Wc2{i} + Step*dWc2;
    Wa1{i+1} = Wa1{i} + Step*dWa1;
    Wa2{i+1} = Wa2{i} + Step*dWa2;
    GAMMA1{i+1} = GAMMA1{i} + Step*dGAMMA1;
    GAMMA2{i+1} = GAMMA2{i} + Step*dGAMMA2;
end


figure(1);
x_plot = cell2mat(x);
plot(t,x_plot);
figure(2);
xm_plot = cell2mat(xm);
plot(t,x_plot - xm_plot);       
figure(3);
Wa1_plot = cell2mat(Wa1);
Wc1_plot = cell2mat(Wc1);
Wa2_plot = cell2mat(Wa2);
Wc2_plot = cell2mat(Wc2);

subplot(1,2,1);
plot(t,Wc1_plot);
subplot(1,2,2);
plot(t,Wa1_plot);

figure(4);
subplot(1,2,1);
plot(t,Wc2_plot);
subplot(1,2,2);
plot(t,Wa2_plot);

u1_plot = cell2mat(u1);
u2_plot = cell2mat(u2);
u1_chuan_plot = cell2mat(u1_chuan);
u2_chuan_plot = cell2mat(u2_chuan);

function a = f_fun(x)
x1 = x(1);
x2 = x(2);
a = [x2 - 2*x1;
    -1/2*x1 - x2 + 1/4*x2*(cos(2*x1) + 2)^2 + 1/4*x2*(sin(4*x1^2) + 2)^2];
end

function a = g1_fun(x)
x1 = x(1);
a = [0;cos(2*x1) + 2];
end

function a = g2_fun(x)
x1 = x(1);
a = [0;sin(4*x1^2) + 2];
end

function a = Y_fun(x)
x1 = x(1);
x2 = x(2);
a = [x2 0;
    x1 0;
    0 x1;
    0 x2;
    0 x2*(cos(2*x1) + 2)^2;
    0 x2*(sin(4*x1^2) + 2)^2]';
end

function a = d_sigma(x)
x1 = x(1);
x2 = x(2);
a = [2*x1 0;
    x2 x1;
    0 2*x2];
end