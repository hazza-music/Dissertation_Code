% Harriet Drury - MSc Digital Audio Engineering Dissertation.
% August 2022

% Splitting the recording into minute long samples. First 16 files of WHU dataset used
for i = 1:16
    info = audioinfo(strcat('C:/Users/Hairybot/Documents/Masters/Dissertation/Dataset/ENF-WHU-Dataset-master/ENF-WHU-Dataset/H1_ref/',index_ref(i).name));

    Fs = info.SampleRate;
    chunkDuration = 60; %60 seconds/ 1 Minute
    numSamplesPerChunk = chunkDuration*Fs;

    chunkCount = 1;
    for startLoc = 1:numSamplesPerChunk:info.TotalSamples
    endLoc = min(startLoc + numSamplesPerChunk - 1, info.TotalSamples);

    y = audioread(strcat('C:/Users/Hairybot/Documents/Masters/Dissertation/Dataset/ENF-WHU-Dataset-master/ENF-WHU-Dataset/H1_ref/',index_ref(i).name), [startLoc endLoc]);
    outFileName = ['H_',int2str(i),'_',int2str(chunkCount),'_ref','.wav'];
    audiowrite(strcat('C:/Users/Hairybot/Documents/Masters/Dissertation/MATLAB/min_sample_ref/',outFileName), y, Fs);
    %audiowrite(outFileName, y, Fs);
    chunkCount = chunkCount + 1;
    end
end