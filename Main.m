clc, clear all
addpath(genpath(pwd)) % to include the functions under utils

% Name of the dataset
datasetName = 'CIFAR100';
% Load features extracted from trained CNN model
LoadFrom = 'E:\Sara_CWS2_Backup\CIFAR-100';
% Load the ECOC matrix -- ECOC matrix can be randomly generated using GenerateCodeWithSpecificHAmmingDistance.m
ECOCP = 'E:\Sara_CWS2_Backup\ECOCMatrices\ECOC_100_300_138_1.mat';

% Train the model for several trials and report the mean and std
NTrials = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load training data and labels
TrainFeatures = struct2cell(load([LoadFrom{md}, '\', datasetName, '_trainingData_avgpool.mat']));
TrainFeatures = TrainFeatures{1};
TrainLabels = struct2cell(load([LoadFrom{md}, '\', datasetName, '_trainingLabel.mat']));
TrainLabels = TrainLabels{1}+1;

% Load testing data and labels
TestFeatures = struct2cell(load([LoadFrom{md}, '\', datasetName, '_testingData_avgpool.mat']));
TestFeatures = TestFeatures{1};
TestLabels = struct2cell(load([LoadFrom{md}, '\', datasetName, '_testingLabel.mat']));
TestLabels = TestLabels{1}+1;

% Load ECOC matrix
ECOC = load(ECOCP);
ECOC = ECOC.ECOC; ECOC(ECOC == 0) = -1;

F = fopen('Results.txt', 'w');
for ECOCTrial = 1 : NTrials
    % training model
    [trainingTime, net] = MultiTask_ECOC_Fast(ECOC, TrainFeatures, TrainLabels);
    
    % testing 
    tic;
    pred = predict(net, TestFeatures);
    [numOfClasses, numOfClassifiers] = size(ECOC);
    predWordCode = pred(:, 1:numOfClassifiers);
    pred = predWordCode*ECOC';
    [~, b] = max(pred, [], 2);
    acc = mean(b(:) == TestLabels(:));
    testingTime = toc';
    
    fprintf(F, 'Dataset: %s - Trial = %d - Testing accuracy = %0.3f - Training Time: %0.3f - Testing Time: %0.3f \n', datasetName, ECOCTrial, acc*100, trainingTime/60, testingTime/60);   
end
fclose(F);