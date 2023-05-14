function [ H, G ] = generateNetwork( K, L, numAntenna )

% ITU-1411

range = 1000; % in meter
txPosition = (rand(K,1) + 1i*rand(K,1))*range;
rxPosition = nan(L,1);
maxDist = 65;
minDist = 2;
N = numAntenna;

for i = 1:K
    j = i;
    while(1)
        rand_dir = randn() + randn()*1i;
        dist = (minDist+(maxDist-minDist)*rand);
        rxPosition(j) = dist*rand_dir/norm(rand_dir) + txPosition(i);
%         norm(rxPosition(j)-txPosition(i)) % print dist of each link
        if real(rxPosition(j))>=0 && real(rxPosition(j))<=range ...
            && imag(rxPosition(j))>=0 && imag(rxPosition(j))<=range
            break
        end
    end
end

% % plot the topology
% figure; 
% hold on;
% for i = 1:K
%     j = i;
%     plot([real(txPosition(i)),real(rxPosition(j))],[imag(txPosition(i)),imag(rxPosition(j))],'k');
% end
% plot(real(txPosition), imag(txPosition),'k.');
% plot(real(rxPosition), imag(rxPosition),'k.');
% % axis([-1 1 -1 1]*1.5); legend('Macro BS','Femto BS','MS');
% xlabel('m'); ylabel('m');

c = 3e8; % speed of light
freq = 2.4e9; % in Hz
wavelength = c/freq; % in meter
Rbp = 4*1.5^2/wavelength;
Lbp = abs(20*log10(wavelength^2/(8*pi*1.5^2)));

PL_dB = nan(L,K); % pathloss in dB
for j = 1:L
    for i = 1:K
        dist = abs(txPosition(i)-rxPosition(j));
        if dist<=Rbp
            PL_dB(j,i) = Lbp + 6 + 20*log10(dist/Rbp);
        else
            PL_dB(j,i) = Lbp + 6 + 40*log10(dist/Rbp); % from j to i
        end
    end
    % Antennta gain
    i = j; PL_dB(j,i) = PL_dB(j,i) - 2.5;
end

PL_dB = PL_dB + randn(L,K)*10; % shadowing

PL = 10.^(-PL_dB/10);

%%

H = nan(L,K); % channel coefficient 
G = nan(L,K); % channel magnitude G(rx,tx)
for i = 1:K
    for j = 1:L
        H(j,i) = sqrt(PL(j,i))*(randn(1)+1i*randn(1))/sqrt(2);
        G(j,i) = abs(H(j,i))^2;
    end
end

end