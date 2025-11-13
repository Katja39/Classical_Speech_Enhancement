%sawtooth_wave();
%triangle_wave();

%Fourier analyse
fs = 1000; % Abtastrate
t = 0:1/fs:1-1/fs;  % 1 Sekunde
f0 = 50; % 50 Hz Signal
y = sin(2*pi*f0*t); % Signal erstellen
Y = fft(y); % DFT (über FFT) berechnen
N = length(y); % Datenlänge für das Plotten
f = (0:N-1)*(fs/N); % Frequenzachse für das Plotten

% Plotten
figure; subplot(2,1,1); plot(t, y);
xlabel('Zeit [s]'); ylabel('Amplitude');
title('Zeitsignal');
subplot(2,1,2); plot(f(1:N/2), abs(Y(1:N/2)));
xlabel('Frequenz [Hz]'); ylabel('Magnitude');
title('Frequenzspektrum');

[y, fs] = audioread('snd1.wav');