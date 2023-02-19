%% Identification for nonzero sum game
clc; clear; close all;
%% Time step
Step = 0.001;
T_end = 10;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
u1 = data;
u2 = data;
xm = data;
Wf = data;
Vf = data;
v = data;
%% Parameter
k = 300;
alpha = 20;
gamma = 5;
beta1 = 0.2;
GAMMA_wf = 0.1*eye(5);
GAMMA_vf = 0.1*eye(2);
%% Initial value
x{1} = [3;-1];
xm{1} = randn([2 1]);
Wf{1} = randn([5 2]);
Vf{1} = randn([2 5]);
v{1} = [0;0];
x_nga_0 = x{1} - xm{1};
%% Simulation
for i = 1:length(t)
    x1 = x{i}(1);
    x2 = x{i}(2);
    u1{i} = -(cos(2*x1)+2)*x2;
    u2{i} = -1/2*(sin(4*x1^2)+2)*x2;
    x_nga = x{i} - xm{i};
    muy = k*(x_nga - x_nga_0) + v{i};
    dxm = Wf{i}'*sigmoid(Vf{i}'*x{i})+g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i} + muy;

    if i == length(t)
        break
    end
    %% Update state
    x{i+1} = x{i} + Step*(f_fun(x{i}) + g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i});
    %% Update v
    v{i+1} = v{i} + Step*((k*alpha +gamma)*x_nga + beta1*sign(x_nga));
    %% Update approximate state xm
    xm{i+1} = xm{i} + Step*(Wf{i}'*sigmoid(Vf{i}'*x{i})+g1_fun(x{i})*u1{i} + g2_fun(x{i})*u2{i} + muy);
    %% Update weight identifier
    Wf{i+1} = Wf{i} + Step*(GAMMA_wf*diag(d_sigmoid(Vf{i}'*xm{i}))*Vf{i}'*dxm*x_nga');
    Vf{i+1} = Vf{i} + Step*(GAMMA_vf*dxm*x_nga'*Wf{i}'*diag(d_sigmoid(Vf{i}'*xm{i})));
end

figure(1);
x_plot = cell2mat(x);
xm_plot = cell2mat(xm);
plot(t,x_plot,t,xm_plot);


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