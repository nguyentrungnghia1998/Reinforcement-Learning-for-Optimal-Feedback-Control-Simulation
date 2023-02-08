%% Identification for affine system
clc; clear; close all;
%% Time step
Step = 0.0001;
T_end = 10;
t = 0:Step:T_end;
data = cell(1,length(t));
%% Variable
x = data;
xm = data;
u = data;
Wf = data;
Vf = data;
v = data;
%% Parameter
k = 40;
alpha = 30;
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
    u{i} = -(cos(2*x1)+2)*x2;
    dx = [-x1+x2;-0.5*x1-0.5*x2*(1-(cos(2*x1)+2)^2)] + [0;cos(2*x1)+2]*u{i};
    x_nga = x{i} - xm{i};
    %% Calculate muy(t)
    muy = k*x_nga - k*x_nga_0 + v{i};
    %% Approximate identifier
    dxm = Wf{i}'*[sigmoid(Vf{i}'*xm{i})] + [0;cos(2*x1)+2]*u{i} + muy;
    %% Calculate project derivation weight
    dWf = GAMMA_wf*diag(d_sigmoid(Vf{i}'*xm{i}))*Vf{i}'*dxm*x_nga';
    dVf = GAMMA_vf*dxm*x_nga'*Wf{i}'*diag(d_sigmoid(Vf{i}'*xm{i}));
    if i == length(t)
        break
    end
    %% Update state
    x{i+1} = x{i} + Step*dx;
    %% Update v(t)
    v{i+1} = v{i} + Step*((k*alpha+gamma)*x_nga + beta1*sign(x_nga));
    %% Update xm(t)
    xm{i+1} = xm{i} + Step*dxm;
    %% Update weights
    Wf{i+1} = Wf{i} + Step*dWf;
    Vf{i+1} = Vf{i} + Step*dVf;
end

x_plot = cell2mat(x);
xm_plot = cell2mat(xm);
plot(t,x_plot,t,xm_plot);


function a = sigmoid(x)
a = 1./(1+exp(-x));
end

function a = d_sigmoid(x)
a = exp(-x)./((1 + exp(-x)).^2);
end