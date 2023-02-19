%% Actor critic for 2 player nonzero sum
clc; clear; close all;
%% Time step
Step = 0.00005;
T_end = 20;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
xm = data;
Wa1 = data;
Wa2 = data;
Wc1 = data;
Wc2 = data;
Wf = data;
Vf = data;
u1 = data;
u2 = data;
v = data;
GAMMA1 = data;
GAMMA2 = data;
%% Parameter
k = 30;
alpha = 20;
gamma = 5;
beta1 = 0.2;
GAMMA_wf = 0.1*eye(5);
GAMMA_vf = 0.2*eye(2);
ka11 = 2;
ka12 = 2;
ka21 = 4;
ka22 = 4;
kc1 = 15;
kc2 = 15;
nuy1 = 0.001;
nuy2 = 0.001;
lamda1 = 0;
lamda2 = 0;
R11 = 2;
R22 = 1;
R12 = 2;
R21 = 1;
Q1 = 2*eye(2);
Q2 = eye(2);
%% Initial value
x{1} = [3;-1];
xm{1} = randn([2 1]);
Wf{1} = randn([5 2]);
Vf{1} = randn([2 5]);
Wc1{1} = [3;3;3] + randn([3 1]);
Wc2{1} = [3;3;3] + randn([3 1]);
Wa1{1} = [3;3;3] + randn([3 1]);
Wa2{1} = [3;3;3] + randn([3 1]);
x_nga_0 = x{1} - xm{1};
v{1} = [0;0];
GAMMA1{1} = 5000*eye(3);
GAMMA2{1} = 5000*eye(3);
%% Simulation
for i = 1:length(t)
    if t(i) <= 10
        noise = sin(5*pi*t(i)) + sin(exp(1)*t(i)) + sin(t(i))^5 + cos(20*t(i))^5 + sin(-1.2*t(i))^2*cos(0.5*t(i));
    else
        noise = 0;
    end
    u1{i} = -1/2*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i};
    u2{i} = -1/2*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i};
    dx = f_fun(x{i}) + g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i};
    x_nga = x{i} - xm{i};
    muy = k*(x_nga - x_nga_0) + v{i};
    dxm = Wf{i}'*sigmoid(Vf{i}'*x{i})+g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i} + muy;
    omega1 = d_sigma(x{i})*dxm;
    omega2 = d_sigma(x{i})*dxm;
    delta_hjb1 = x{i}'*Q1*x{i} + u1{i}'*R11*u1{i} + u2{i}'*R12*u2{i} + Wc1{i}'*omega1;
    delta_hjb2 = x{i}'*Q2*x{i} + u1{i}'*R21*u1{i} + u2{i}'*R22*u2{i} + Wc2{i}'*omega2;
    dWf = GAMMA_wf*diag(d_sigmoid(Vf{i}'*xm{i}))*Vf{i}'*dxm*x_nga';
    dVf = GAMMA_vf*dxm*x_nga'*Wf{i}'*diag(d_sigmoid(Vf{i}'*xm{i}));
    dv = (k*alpha + gamma)*x_nga + beta1*sign(x_nga);
    dWc1 = -kc1*GAMMA1{i}*omega1/(1+nuy1*omega1'*GAMMA1{i}*omega1)*delta_hjb1;
    dWc2 = -kc2*GAMMA2{i}*omega2/(1+nuy2*omega2'*GAMMA2{i}*omega2)*delta_hjb2;
    dGAMMA1 = -kc1*(lamda1*GAMMA1{i} + GAMMA1{i}*(omega1*omega1')/(1+nuy1*omega1'*GAMMA1{i}*omega1)*GAMMA1{i});
    dGAMMA2 = -kc2*(lamda2*GAMMA2{i} + GAMMA2{i}*(omega2*omega2')/(1+nuy2*omega2'*GAMMA2{i}*omega2)*GAMMA2{i});
    Ea1 = (d_sigma(x{i})*g1_fun(x{i})*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*(Wa1{i} - Wc1{i}))*delta_hjb1 + (d_sigma(x{i})*g1_fun(x{i})*R11^-1*R12*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wa1{i} - d_sigma(x{i})*g1_fun(x{i})*R11^-1*g1_fun(x{i})'*d_sigma(x{i})'*Wc2{i})*delta_hjb2;
    Ea2 = (d_sigma(x{i})*g2_fun(x{i})*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*(Wa2{i} - Wc2{i}))*delta_hjb2 + (d_sigma(x{i})*g2_fun(x{i})*R22^-1*R21*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wa2{i} - d_sigma(x{i})*g2_fun(x{i})*R22^-1*g2_fun(x{i})'*d_sigma(x{i})'*Wc1{i})*delta_hjb1;
    dWa1 = -ka11/sqrt(1+omega1'*omega1)*Ea1 - ka12*(Wa1{i} - Wc1{i});
    dWa2 = -ka21/sqrt(1+omega2'*omega2)*Ea2 - ka22*(Wa2{i} - Wc2{i});
%     dWa1 = -ka12*(Wa1{i} - Wc1{i});
%     dWa2 = -ka22*(Wa2{i} - Wc2{i});

    if i == length(t)
        break
    end
    %% Update state
    x{i+1} = x{i} + Step*(f_fun(x{i}) + g1_fun(x{i})*(u1{i}+noise) + g2_fun(x{i})*(u2{i}+noise));
    %% Update dxm
    xm{i+1} = xm{i} + Step*(Wf{i}'*sigmoid(Vf{i}'*x{i})+g1_fun(x{i})*(u1{i}+noise) + g2_fun(x{i})*(u2{i}+noise) + muy);
    v{i+1} = v{i} + Step*dv;
    %% Update identifier weight
    Wf{i+1} = Wf{i} + Step*dWf;
    Vf{i+1} = Vf{i} + Step*dVf;
    %% Update actor-critic
    Wc1{i+1} = Wc1{i} + Step*dWc1;
    Wc2{i+1} = Wc2{i} + Step*dWc2;
    Wa1{i+1} = Wa1{i} + Step*dWa1;
    Wa2{i+1} = Wa2{i} + Step*dWa2;
    GAMMA1{i+1} = GAMMA1{i} + Step*dGAMMA1;
    GAMMA2{i+1} = GAMMA2{i} + Step*dGAMMA2;
    if min(eig(GAMMA1{i+1})) < 5
        GAMMA1{i+1} = 5000*eye(3);
    end
    if min(eig(GAMMA2{i+1})) < 5
        GAMMA2{i+1} = 5000*eye(3);
    end
    
end

figure(1);
x_plot = cell2mat(x);
plot(t,x_plot);

figure(2);
Wa1_plot = cell2mat(Wa1);
Wc1_plot = cell2mat(Wc1);
subplot(1,2,1);
plot(t,Wa1_plot);
subplot(1,2,2);
plot(t,Wc1_plot);

figure(3);
Wa2_plot = cell2mat(Wa2);
Wc2_plot = cell2mat(Wc2);
subplot(1,2,1);
plot(t,Wa2_plot);
subplot(1,2,2);
plot(t,Wc2_plot);

function a = f_fun(x)
x1 = x(1);
x2 = x(2);
a = [x2-2*x1;
    -1/2*x1-x2 + 1/4*x2*(cos(2*x1)+2)^2 + 1/4*x2*(sin(4*x1^2) + 2)^2];
end

function a = g1_fun(x)
x1 = x(1);
a = [0;cos(2*x1)+2];
end

function a = g2_fun(x)
x1 = x(1);
a = [0;sin(4*x1^2)+2];
end

function a = sigmoid(x)
a = 1./(1+exp(-x));
end

function a = d_sigmoid(x)
a = exp(-x)./((1+exp(-x)).^2);
end

function a = d_sigma(x)
x1 = x(1);
x2 = x(2);
a = [2*x1 0;
    x2 x1;
    0 2*x2];
end