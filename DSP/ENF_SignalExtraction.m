% Harriet Drury - MSc Digital Audio Engineering Dissertation.
% August 2022

% Band limiting audio data and creating spectrogram visualisations for
% Keras API

clear; close; clc;
set(0,'DefaultFigureWindowStyle','docked')  % 'normal' to un-dock. stops figures appearing on screen when generated (useful!)

%% parameter setting and initialization
FS                       = 999; % constant sampling frequency
HARMONIC_INDEX           = [2,3,4,5,6,7]; % constant value for ENF harmonic processing
fc                       = 50*HARMONIC_INDEX; % nominal frequencies at each harmonic
bound                    = 0.1*HARMONIC_INDEX; % tolerable IF deviations at each harmonic
filter_length            = 256;
[BPF_coeffs, coeffs_2nd] = func_BPF(filter_length);

% NOTE :- Data directories
% C:/Users/Hairybot/Documents/Masters/Dissertation/MATLAB/AudioSampleData/min_sample/*.wav
% C:/Users/Hairybot/Documents/Masters/Dissertation/MATLAB/AudioSampleData/min_sample_ref/*.wav

fds = fileDatastore('C:/Users/Hairybot/Documents/Masters/Dissertation/MATLAB/AudioSampleData/min_sample_ref/*.wav', 'ReadFcn', @importdata);

fullFileNames = fds.Files;
sort(fullFileNames);

numFiles = length(fullFileNames);

% Loop over all files reading them in and plotting them.
% NOTE:- SAMPLE H_16_13 SKIPPED. SKIP THE REF ONE TOO
% C:\Users\Hairybot\Documents\Masters\Dissertation\MATLAB\AudioSampleData\min_sample\H_16_13.wav
% C:\Users\Hairybot\Documents\Masters\Dissertation\MATLAB\AudioSampleData\min_sample_ref\H_16_13_ref.wav

for k = 1 : numFiles
    name = fullFileNames{k};
    if strcmp(name,'C:\Users\Hairybot\Documents\Masters\Dissertation\MATLAB\AudioSampleData\min_sample_ref\H_16_13_ref.wav') == 1
        fprintf('Skipping this one...\n');
    else
    fprintf('Now reading file %s\n', fullFileNames{k});
     [audio, fs_audio] = audioread(fullFileNames{k});
     audio             = audio(:,1);
     audio             = audio';
     raw_wave          = resample(audio, FS, fs_audio);
     N                 = length(raw_wave); % Number of samples
     %% bandpass filtering
     input             = filtfilt(BPF_coeffs,1,raw_wave);

     %% Spectrogram Generation

     % Reference Settings: Nspec = 1000, wspec = hamming(Nspec), Noverlap =
     % Nspec/2
     Nspec = 1000;
     wspec = hamming(Nspec);
     Noverlap = Nspec/2;

     figure(k)
     hold on
     spectrogram(input, wspec,Noverlap,2048,FS, 'yaxis');
     set(gca, 'Visible', 'off')
     colorbar('off');
     set(gcf, 'Position', get(0, 'Screensize'));

     set(gcf, 'color', 'none');    
     set(gca, 'color', 'none');
     
     % add '_ref' to end of outFileName when working with ref
     outFileName = ['H_fig_',int2str(k),'_ref','.png'];
     % outFileName = ['H_fig_',int2str(k),'.png'];
     %saveas(gcf,outFileName);
      
     exportgraphics(gcf,outFileName,...   % since R2020a
    'ContentType','vector',...
    'BackgroundColor','none')

     hold off
    end
end