function sawtooth_wave(fs, f0, n)
 % Erstellen von Sägezahnwelle durch Fourier-Reihe
 % fs: Abtastrate, f0: Grundfrequenz, n: Anzahl Harmonischen
 if ~exist('fs','var'), fs=8000; end % default-Wert für fs
 if ~exist('f0','var'), f0=500; end % default-Wert für f0
 if ~exist('n','var'), n=20; end  % default-Wert für n
 t= 0:1/fs:1-1/fs;
 y= zeros(size(t));
 
 % Addieren der ersten n Harmonischen
 for k = 1:n
 y = y + ((-1)^(k+1) / k) * sin(2*pi*k*f0*t);
 end
 
 y = y * (2/pi); 
pn= ceil(2*fs/f0);      
plot(t(1:pn), y(1:pn));
 xlabel('Zeit [s]'); ylabel('Amplitude');
 title('Sägezahn durch Fourier-Reihe');
 end