function main
% Sets parameters
n=10000;        %Time step
N=1;           %Number of trajectories
T=1;
tau=0.5;
% Interest rate
r = 0.03;
% Spot parameters
mu = 0.1;
sigma_S=0.3;
% Convenience yield parameters
kappa=3;
alpha=0.04;
sigma_C=0.5;
lambda = 0.021;
% Correlation
rho=0.6;
% Initial values
S0=100;
C0=alpha;
%
% Simulation
[Spot, CY] = Schwartz_rand(n, N, T, S0, C0, mu, sigma_S, kappa, alpha, sigma_C, rho );
% Draws spot
figure();
plot(0:T/n:T, Spot);
title('Random trajectories for Spot');
xlabel( 'Time');
ylabel( 'Spot');
% Draws convenience yield
figure();
plot(0:T/n:T, CY);
title('Random trajectories for Convenience yield');
xlabel( 'Time');
ylabel( 'Convenience yield');
% Pricing du foward à maturité glissante T
figure();
F = Schwartz_Forward(Spot, CY, tau, r, sigma_S, kappa, alpha, sigma_C, rho, lambda);
plot(0:T/n:T, F);
title('Random trajectories for Forward price');
xlabel( 'Time');
ylabel( 'Forward price');
%
% Filtering
dt = T/n;    %Pas de temps de l'échantillonage
y = log(F);  %Observed signal
logS_f = zeros(1, n+1); %Filtred signals
C_f = zeros(1, n+1);
P_f = zeros(2,2,n+1);
% Proxy initiaux avant filtrage
% logS_f = logF
% C = 0 car en moyenne nul
% P_f = la matrice de variance covariance de dlogS et dC
logS_f(1) = log(F(1));  % Bon proxy avant tout filtrage
C_f(1) = 0;
P_f(:,:,1) = [ sigma_S^2, rho*sigma_S*sigma_C; rho*sigma_S*sigma_C, sigma_C^2]*dt;
% Equation de mesure y_t = Z alpha_t + d + N(0,H)
Z = [1, -(1-exp(-kappa*tau))/kappa];
alpha_c = alpha - lambda/kappa;
d = ( (r-alpha_c+sigma_C*sigma_C/(2*kappa*kappa)- sigma_S*sigma_C*rho/kappa)*tau) + sigma_C*sigma_C*(1-exp(-2*kappa*tau))/(4*kappa*kappa*kappa) + (alpha_c*kappa + sigma_S*sigma_C*rho - sigma_C*sigma_C/kappa)*(1-exp(-kappa*tau))/(kappa*kappa);
H = 0.001*sigma_S^2;
% Equation de transition de alpha_t
% alpha_{t+dt} = T alpha_t + C + R N(0, Q)
T = [1, -dt; 0, 1-kappa*dt];
C = dt*[mu-sigma_S*sigma_S/2; kappa*alpha];
R = sqrt(dt)*[sigma_S, 0; sigma_C*sqrt(1-rho*rho), rho*sigma_C];
Q = eye(2);
for i=1:n
    alpha_f = [logS_f(i); C_f(i)];
    %Prediction (pred = t|(t-1) )
    alpha_pred = T*alpha_f + C;
    P_pred = R*Q*transpose(R) + T*P_f(:,:,i)*transpose(T);
    %Innovation
    y_pred = Z*alpha_pred + d;
    error = y(i+1)-y_pred;
    F = Z*P_pred*transpose(Z) + H;
    % Mise à jour
    alpha_f = alpha_pred + P_pred*transpose(Z)*(F^(-1))*error;
    P_f(:,:,i+1) = (eye(2,2) - P_pred *transpose(Z)*(F^(-1))*Z)*P_pred;
    % Copie
    logS_f(i+1) = alpha_f(1);
    C_f(i+1) = alpha_f(2);
end
S_f = exp(logS_f);
figure();
logS_f_std_dev = sqrt(transpose(squeeze(P_f(1,1,:))));
plot(0:T/n:T, [exp(logS_f - 1.96*logS_f_std_dev); S_f; exp(logS_f + 1.96*logS_f_std_dev)]);
title('Filtered trajectories for Spot');
xlabel( 'Time');
ylabel( 'Spot');
% Draws convenience yield
figure();
C_f_std_dev = sqrt(transpose(squeeze(P_f(2,2,:))));
plot(0:T/n:T, [C_f - 1.96*C_f_std_dev; C_f; C_f + 1.96*C_f_std_dev]);
title('Filtered trajectories for Convenience yield');
xlabel( 'Time');
ylabel( 'Convenience yield');
end

function F = Schwartz_Forward(S, C, tau, r, sigma_S, kappa, alpha, sigma_C, rho, lambda)
alpha_c = alpha - lambda/kappa;
B = ( (r-alpha_c+sigma_C*sigma_C/(2*kappa*kappa)- sigma_S*sigma_C*rho/kappa)*tau) + sigma_C*sigma_C*(1-exp(-2*kappa*tau))/(4*kappa*kappa*kappa) + (alpha_c*kappa + sigma_S*sigma_C*rho - sigma_C*sigma_C/kappa)*(1-exp(-kappa*tau))/(kappa*kappa);
F = S.*exp( -C * (1-exp(-kappa*tau))/kappa + B);
end

function [S, C] = Schwartz_rand(n, N, T, S0, C0, mu, sigma_S, kappa, alpha, sigma_C, rho )
dt = (T/n)*ones(N, n);
time = cumsum( dt, 2);
S = zeros(N, n+1);
C = zeros(N, n+1);
% Draws C
dW_C = sqrt(T/n)*randn(N, n);
Gaussian_process = zeros( N, n);
Gaussian_process = exp(-kappa*time).*cumsum( exp( kappa*time ).*dW_C, 2 );
C(:, 1) = C0*ones( N, 1);
C(:, 2:(n+1)) = alpha*ones(N, n) + (C0-alpha)*exp( -kappa*time ) + sigma_C*Gaussian_process;
% Draws S
dW_S = rho*dW_C + sqrt(1-rho*rho)*sqrt(T/n)*randn(N, n);
W_S = cumsum(dW_S, 2);
S(:, 1) = S0*ones(N, 1);
S(:, 2:(n+1)) = S0*exp( (mu - sigma_S*sigma_S/2)*time + sigma_S*W_S - (T/n)*cumsum(C(:,2:(n+1)), 2) );
end