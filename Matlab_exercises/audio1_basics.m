%[y, fs] = audioread('snd1.wav'); 
%y_rev = flipud(y);      % reverse
%wavplay(y_rev, fs);              

%Was passiert, wenn Sie die Audiodatei snd2.wav mit abweichender Abtastrate abspielen?
%[y,fs]= audioread('snd2.wav');
%wavplay(y,fs);
%wavplay(y,0.7*fs); % Abtastrate 70% vom Original,  deeper
%wavplay(y,1.3*fs);  % Abtastrate 130% vom Original, higher

%Was passiert, wenn Sie y mit einem Skalar multiplizieren?
%[y,fs]= audioread('snd2.wav');
%wavplay(0.5*y,fs); 
%wavplay(0.125*y,fs); %leiser
%wavplay(2.0*y,fs); %lauter

%y_min = min(y(:));
%y_max = max(y(:));

%peak = max(abs(y(:)));      % peak amplitude
%peak_dBFS = 20*log10(peak); % peak level in dBFS

%Was bezweckt diese Funktion? - Quantisierung
%plot(quantopt(sin(linspace(0,2*pi,100)),10))
%plot(quantopt(sin(linspace(0,2*pi,100)),5))
%plot(quantopt(sin(linspace(0,2*pi,100)),2))

% Vergleich von quantopt und quantize (2-Bit)

%figure;
%hold on; grid on;
%
%plot(quantopt(sin(linspace(0,2*pi,100)),2), 'r', 'LineWidth', 2);
%plot(quantize(sin(linspace(0,2*pi,100)),2), 'b--', 'LineWidth', 2);
%
%xlabel('Abtastindex');
%ylabel('Amplitude');
%title('Vergleich von quantopt und quantize (2-Bit-Quantisierung)');
%
%legend('quantopt (rot)', 'quantize (blau gestrichelt)');
%axis tight;

%Bittiefe
%[y,fs]= audioread('snd3.wav');
%wavplay(quantize(y,16),fs); %cd quality
%wavplay(quantize(y,8),fs);  %telefon quality
%wavplay(quantize(y,6),fs);  %verzerrt
%wavplay(quantize(y,4),fs);  %kaum noch zu verstehen

%Abtaständerung, Klang wird immer dumpfer, Downsampling, Ursprüngliche
%Abtastrate/r
%Aliasing tritt auf
%[y,fs]= audioread('snd2.wav');
%r=3; wavplay(y(1:r:end),fs/r);
%r=6; wavplay(y(1:r:end),fs/r);
%r=9; wavplay(y(1:r:end),fs/r);

%besserer Klang, wenig Störgeräusch
%durch Tiefpassfilter, Antialising
%r=3; wavplay(resample(y,fs,fs*r),fs/r); 
%r=6; wavplay(resample(y,fs,fs*r),fs/r); 
%r=9; wavplay(resample(y,fs,fs*r),fs/r);

%Verzerrung
%[y, fs] = audioread('snd5.wav');
%wavplay(y,fs); 
%y_dist = y .^ 3;
%wavplay(y_dist,fs); 

%Echo
%[y, fs] = audioread('snd5.wav');
%
%r = round(0.3 * fs);    % Verzögerung 0.3
%a = 0.6;                % Lautstärke
%
%y_echo = y;             
%y_echo(r+1:end) = y_echo(r+1:end) + a * y(1:end-r); 
%
%wavplay(y_echo,fs);



