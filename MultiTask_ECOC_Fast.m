function [tim_Train, net] = MultiTask_ECOC_Fast(ECOC, TrainFeatures, TrainLabels)

[~, numOfClassifiers] = size(ECOC);

% Map GT to class code
GT_EcocMapped = zeros(length(TrainLabels), numOfClassifiers);
for i = 1 : length(TrainLabels)
    GT_EcocMapped(i, :) = ECOC(TrainLabels(i), :);
end

%% Create architecture
layers = [imageInputLayer([1 1 2048], 'Name', 'input')
    dropoutLayer('Name', 'drop0')
    fullyConnectedLayer(500, 'Name', 'fc1')
    reluLayer('Name','relu1')
    dropoutLayer('Name', 'drop1')
    fullyConnectedLayer(50, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    dropoutLayer('Name', 'drop2')];
lgraph = layerGraph(layers);

f3 = fullyConnectedLayer_SubSplit(10*numOfClassifiers, 'Name', 'fc3', 'splitN', numOfClassifiers);
f3.Bias = zeros(f3.OutputSize, 1);
f3.BiasL2Factor = 0;
f3.BiasLearnRateFactor = 0;

lgraph = addLayers(lgraph, f3);
lgraph = connectLayers(lgraph, 'drop2', 'fc3');

lgraph = addLayers(lgraph, reluLayer('Name', 'relu3'));
lgraph = connectLayers(lgraph, 'fc3', 'relu3');

f4 = fullyConnectedLayer_FullSplit(numOfClassifiers, 'Name', 'fc4', 'splitN', numOfClassifiers);
f4.Bias = zeros(f4.OutputSize, 1);
f4.BiasL2Factor = 0;
f4.BiasLearnRateFactor = 0;

lgraph = addLayers(lgraph, f4);
lgraph = connectLayers(lgraph, 'relu3', 'fc4');

lgraph = addLayers(lgraph, tanhLayer('Name', 'tanh'));
lgraph = connectLayers(lgraph, 'fc4', 'tanh');

lgraph = addLayers(lgraph, regressionLayer('Name', 'Loss'));
lgraph = connectLayers(lgraph, 'tanh', 'Loss');

options = trainingOptions('rmsprop', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true);

tic;
[net, ~] = trainNetwork(TrainFeatures, GT_EcocMapped, lgraph, options);
tim_Train = toc';


end


