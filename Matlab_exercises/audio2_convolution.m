%ir= [1, zeros(1,5e3), 0.5, zeros(1,8e3), 0.5, zeros(1,9e3), 0.25];
%ir = [0.5 1 0.5]; 
%[y,fs]= audioread('FHE_Theaterraum_H-IR.wav');
%[x,fs]= audioread('snd4.wav');

%wavplay(y,fs);

%only mono
%y_echo= conv(y,ir);

%y_left  = conv(y(:,1), ir);
%y_right = conv(y(:,2), ir);

%y_falt = [y_left, y_right];

%wavplay(y_falt,fs); %räumlicher Klang, länger

%falten mit anderem sound

y_left  = conv(y(:,1), x);
y_right = conv(y(:,2), x);
y_falt = [y_left, y_right];

wavplay(y_falt,fs);