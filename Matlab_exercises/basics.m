%Plotten Sie eine Cosinus Funktion zwischen -2π und 2π mit einer Auflösung von 100 Punkte / π!
%mit einem Hann-Fenster gewichtet

n = 400; % 100 points per π → 400 points
x = linspace(-2*pi, 2*pi, n);  

% Compute the cosine values
y = cos(x);

% Create Hann window of same length
w = hann(n)';              

y_win = y .* w;

% Plot original and windowed cosine for comparison
figure;
plot(x, y, 'b--', 'LineWidth', 1.2); hold on;
plot(x, y_win, 'r', 'LineWidth', 1.8);
grid on;

% Labels
xlabel('x');
ylabel('Amplitude');
title('Cosine Function Weighted by Hann Window');
legend('Original cos(x)', 'Windowed cos(x)');