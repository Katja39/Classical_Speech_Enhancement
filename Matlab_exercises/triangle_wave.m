function triangle_wave(fs, f0, n)
% Erstellen einer Dreieckswelle durch Fourier-Reihe
% fs: Abtastrate [Hz]
% f0: Grundfrequenz [Hz]
% n:  Anzahl der Harmonischen

    if ~exist('fs','var'), fs = 2000; end
    if ~exist('f0','var'), f0 = 100; end
    if ~exist('n','var'),  n = 20; end
    
    t = 0:1/fs:1-1/fs;
    y = zeros(size(t));

    % Nur ungerade harmonische - bei sawtooth alle harmonischen
    for k = 0:n-1
        harm = 2*k + 1;
        y = y - ((-1)^k / harm^2) * sin(2*pi*harm*f0*t);
    end

    y = y * (2/pi);

    % Anzahl Perioden
    pn = ceil(2*fs/f0);

    % Plot
    plot(t(1:pn), y(1:pn));
    xlabel('Zeit [s]');
    ylabel('Amplitude');
    title('Dreieckswelle durch Fourier-Reihe');
    grid on;
end
