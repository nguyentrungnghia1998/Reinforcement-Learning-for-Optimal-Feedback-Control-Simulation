%% ADP Tracking for Euler Lagrange
clc; clear; close all
%% Time step
Step = 0.0005;
T_end = 60;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
u = data;
Wc = data;
Wa = data;
GAMMA = data;
xd = data;
%% Parameters
p1 = 3.473;
p2 = 0.196;
p3 = 0.242;
ka1 = 5;
ka2 = 0;
kc = 1.25;
lamda = 0;
Q = diag([10 10 2 2]);
Q_ = blkdiag(Q,zeros(4));
R = eye(2);
nuy = 0.005;
%% Initial Value
x{1} = [1.8;1.6;0;0];
Wc{1} = 10 + randn([23 1]);
Wa{1} = 6 + randn([23 1]);
GAMMA{1} = 2000*eye(23);
%% Simulation
for i = 1:length(t)
    if t(i)<=50
        noise = [2.55*tanh(2*t(i))*(20*sin(sqrt(232)*pi*t(i))*cos(sqrt(20)*pi*t(i))+6*sin(18*exp(2)*t(i))+20*cos(40*t(i))*cos(21*t(i)));
            0.01*tanh(2*t(i))*(20*sin(sqrt(132)*pi*t(i))*cos(sqrt(10)*pi*t(i))+6*cos(8*exp(1)*t(i))+20*cos(10*t(i))*cos(11*t(i)))];
    else
        noise = [0;0];
    end
    xd{i} = [0.5*cos(2*t(i));1/3*cos(3*t(i));-sin(2*t(i));-sin(3*t(i))];
    e = x{i} - xd{i};
    psi = [e;xd{i}];
    G = [g_fun(x{i});zeros(4,2)];
    u{i} = -1/2*pinv(R)*G'*d_sigma(psi)'*Wa{i} + pinv(g_fun(xd{i}))*(hd_fun(xd{i}) - f_fun(xd{i}));
    muy = -1/2*pinv(R)*G'*d_sigma(psi)'*Wa{i};
    dx = f_fun(x{i}) + g_fun(x{i})*u{i};
    %% Calculate omega
    omega = d_sigma(psi)*[dx - hd_fun(xd{i});hd_fun(xd{i})];
    delta_hjb = Wc{i}'*omega + psi'*Q_*psi + muy'*R*muy;
    dWc = -kc*GAMMA{i}*omega*1/(1+nuy*omega'*GAMMA{i}*omega)*delta_hjb;
    dGAMMA = -kc*(-lamda*GAMMA{i} + GAMMA{i}*(omega*omega')/(1+nuy*omega'*GAMMA{i}*omega)*GAMMA{i});
    dWa = -ka1*(Wa{i} - Wc{i}) - ka2*Wa{i};

    if i == length(t)
        break
    end
    %% Update state
    x{i+1} = x{i} + Step*(f_fun(x{i}) + g_fun(x{i})*(u{i} + noise));
    %% Update weight
    Wc{i+1} = Wc{i} + Step*dWc;
    Wa{i+1} = Wa{i} + Step*dWa;
    GAMMA{i+1} = GAMMA{i} + Step*dGAMMA;
%     if min(eig(GAMMA{i+1}))<20
%         GAMMA{i+1} = GAMMA{1};
%     end
end

x_plot = cell2mat(x);
xd_plot = cell2mat(xd);
figure(1);
plot(t,x_plot);

figure(2);
plot(t,x_plot - xd_plot);

Wc_plot = cell2mat(Wc);
Wa_plot = cell2mat(Wa);
figure(3);
plot(t,Wc_plot);
figure(4);
plot(t,Wa_plot);
function a = f_fun(x)
p1 = 3.473;
p2 = 0.196;
p3 = 0.242;
x2 = x(2);
x3 = x(3);
x4 = x(4);
M = [p1+2*p3*cos(x2) p2+p3*cos(x2);
    p2+p3*cos(x2) p2];
Vm = [-p3*sin(x2)*x4 -p3*sin(x2)*(x3+x4);
    p3*sin(x2)*x3 0];
Fd = diag([5.3 1.1]);
Fs = [8.45*tanh(x3);2.35*tanh(x4)];
a = [x3;x4;pinv(M)*(-Vm-Fd)*[x3;x4]-Fs];
end

function a = g_fun(x)
p1 = 3.473;
p2 = 0.196;
p3 = 0.242;
x2 = x(2);
M = [p1+2*p3*cos(x2) p2+p3*cos(x2);
    p2+p3*cos(x2) p2];
a = [0 0;0 0;pinv(M)];
end

function a = hd_fun(xd)
xd1 = xd(1);
xd2 = xd(2);
xd3 = xd(3);
xd4 = xd(4);
a = [xd3;xd4;-4*xd1;-9*xd2];
end

function a = d_sigma(psi)
a = [2*psi(1) 0 0 0 0 0 0 0;
    0 2*psi(2) 0 0 0 0 0 0;
    psi(3) 0 psi(1) 0 0 0 0 0;
    psi(4) 0 0 psi(1) 0 0 0 0;
    0 psi(3) psi(2) 0 0 0 0 0;
    0 psi(4) 0 psi(2) 0 0 0 0;
    2*psi(1)*psi(2)^2 2*psi(2)*psi(1)^2 0 0 0 0 0 0;
    2*psi(1)*psi(5)^2 0 0 0 2*psi(5)*psi(1)^2 0 0 0;
    2*psi(1)*psi(6)^2 0 0 0 0 2*psi(6)*psi(1)^2 0 0;
    2*psi(1)*psi(7)^2 0 0 0 0 0 2*psi(7)*psi(1)^2 0;
    2*psi(1)*psi(8)^2 0 0 0 0 0 0 2*psi(8)*psi(1)^2;
    0 2*psi(2)*psi(5)^2 0 0 2*psi(5)*psi(2)^2 0 0 0;
    0 2*psi(2)*psi(6)^2 0 0 0 2*psi(6)*psi(2)^2 0 0;
    0 2*psi(2)*psi(7)^2 0 0 0 0 2*psi(7)*psi(2)^2 0;
    0 2*psi(2)*psi(8)^2 0 0 0 0 0 2*psi(8)*psi(2)^2;
    0 0 2*psi(3)*psi(5)^2 0 2*psi(5)*psi(3)^2 0 0 0;
    0 0 2*psi(3)*psi(6)^2 0 0 2*psi(6)*psi(3)^2 0 0;
    0 0 2*psi(3)*psi(7)^2 0 0 0 2*psi(7)*psi(3)^2 0;
    0 0 2*psi(3)*psi(8)^2 0 0 0 0 2*psi(8)*psi(3)^2;
    0 0 0 2*psi(4)*psi(5)^2 2*psi(5)*psi(4)^2 0 0 0;
    0 0 0 2*psi(4)*psi(6)^2 0 2*psi(6)*psi(4)^2 0 0;
    0 0 0 2*psi(4)*psi(7)^2 0 0 2*psi(7)*psi(4)^2 0;
    0 0 0 2*psi(4)*psi(8)^2 0 0 0 2*psi(8)*psi(4)^2]*1/2;
end