%% Actor - critic with Identification
clc; clear; close all;
%% Time step
Step = 0.001;
T_end = 40;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
xm = data;
u = data;
Wf = data;
Vf = data;
v = data;
Wc = data;
Wa = data;
GAMMA = data;
%% Parameter
k = 800;
alpha = 300;
gamma = 5;
beta1 = 0.2;
GAMMA_wf = 0.1*eye(5);
GAMMA_vf = 0.1*eye(2);
gamma_upper = 5000;
gamma_lower = 50;
kc = 2;
ka1 = 1;
ka2 = 5;
nuy = 0.005;
Q = eye(2);
R = 1;
%% Initial value
x{1} = [3;-1];
xm{1} = randn([2 1]);
% xm{1} = x{1};
Wf{1} = randn([5 2]);
Vf{1} = randn([2 5]);
v{1} = [0;0];
x_nga_0 = x{1} - xm{1};
GAMMA{1} = gamma_upper*eye(3);
Wc{1} = [0.5 0 1]' + randn([3 1]);
% Wc{1} = [0.2249    0.4431    0.8652]';
% Wc{1} = [0.5 0 1]';
% Wa{1} = [0.5 0 1]' + randn([3 1]);
Wa{1} = [1.6436   -0.9147    1.1798]';
%% Simulation
for i = 1:length(t)
    x1 = x{i}(1);
    x2 = x{i}(2);
    if t(i)<=16
        noise = sin(t(i))^2*cos(t(i)) + sin(2*t(i))^2*cos(0.1*t(i)) + sin(-1.2*t(i))^2*cos(0.5*t(i)) + sin(t(i))^5;
    else
        noise = 0;
    end
    u{i} = -1/2*R^-1*[0 cos(2*x1)+2]*d_sigma(x{i})'*Wa{i};
%     u{i} = -(cos(2*x1)+2)*x2;
    dx = [-x1 + x2;-0.5*x1-0.5*x2*(1-(cos(2*x1)+2)^2)] + [0;cos(2*x1)+2]*(u{i} + noise);
    %% Calculate MUY
    x_nga = x{i} - xm{i};
    muy = k*x_nga - k*x_nga_0 + v{i};
    %% Identifier
    dxm = Wf{i}'*sigmoid(Vf{i}'*xm{i}) + [0;cos(2*x1) + 2]*u{i} + muy;
    %% Derivative weight identifier
    dWf = GAMMA_wf*diag(d_sigmoid(Vf{i}'*xm{i}))*Vf{i}'*dxm*x_nga';
    dVf = GAMMA_vf*dxm*x_nga'*Wf{i}'*diag(d_sigmoid(Vf{i}'*xm{i}));
    %% DeltaHJB
    omega = d_sigma(x{i})*dxm;
    delta = Wc{i}'*omega + x{i}'*Q*x{i} + u{i}'*R*u{i};
    dWc = -kc*GAMMA{i}*omega/(1+nuy*omega'*GAMMA{i}*omega)*delta;
    dGAMMA = -kc*GAMMA{i}*(omega*omega')*1/(1+nuy*omega'*GAMMA{i}*omega)*GAMMA{i};
    G = [0;cos(2*x1) + 2]*R*[0 cos(2*x1) + 2];
    dWa = -ka1/sqrt(1+omega'*omega)*d_sigma(x{i})*G*d_sigma(x{i})'*(Wa{i} - Wc{i})*delta - ka2*(Wa{i} - Wc{i});
    
    if i == length(t)
        break
    end
    
    %% Update state
    x{i+1} = x{i} + Step*dx;
    %% Update v(t)
    v{i+1} = v{i} + Step*((k*alpha + gamma)*x_nga + beta1*sign(x_nga));
    %% Update weight Wf, Vf
    Wf{i+1} = Wf{i} + Step*dWf;
    Vf{i+1} = Vf{i} + Step*dVf;
    %% Update weight Wc, Wc
    Wc{i+1} = Wc{i} + Step*dWc;
    Wa{i+1} = Wa{i} + Step*dWa;
    %% Update GAMMA
    GAMMA{i+1} = GAMMA{i} + Step*dGAMMA;
    if min(eig(GAMMA{i+1})) <= gamma_lower
        GAMMA{i+1} = gamma_upper*eye(3);
    end
    %% Update identifier
    xm{i+1} = xm{i} + Step*(Wf{i}'*sigmoid(Vf{i}'*xm{i}) + [0;cos(2*x1) + 2]*(u{i}+noise) + muy);
end

figure(1);
x_plot = cell2mat(x);
plot(t,x_plot);
figure(2);
Wa_plot = cell2mat(Wa);
plot(t,Wa_plot);
figure(3);
Wc_plot = cell2mat(Wc);
plot(t,Wc_plot);

function a = sigmoid(x)
a = 1./(1+exp(-x));
end

function a = d_sigmoid(x)
a = exp(-x)./((1 + exp(-x)).^2);
end

function a = d_sigma(x)
x1 = x(1);
x2 = x(2);
a = [2*x1 0;
    x2 x1;
    0 2*x2];
end