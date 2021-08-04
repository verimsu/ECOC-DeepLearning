function [tim_Train, net] = MultiTask_ECOC_WithEmbedding_Fast(ECOC, TrainFeatures, TrainLabels)

[numOfClasses, numOfClassifiers] = size(ECOC);

% Map GT to class code
GT_EcocMapped = zeros(length(TrainLabels), numOfClassifiers+numOfClasses);
for i = 1 : length(TrainLabels)
    class_OHE = zeros(numOfClasses, 1);
    class_OHE(double(TrainLabels(i))) = 1;
    GT_EcocMapped(i, :) = [ECOC(TrainLabels(i), :), class_OHE'];
end

%% Create archeticture
layers = [imageInputLayer([1 1 2048], 'Name', 'input')
    dropoutLayer('Name', 'drop0')
    fullyConnectedLayer(500, 'Name', 'fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(50, 'Name', 'fc2')
    reluLayer('Name', 'relu2')];
lgraph = layerGraph(layers);

f3 = fullyConnectedLayer_SubSplit(10*numOfClassifiers, 'Name', 'fc3', 'splitN', numOfClassifiers);
f3.Bias = zeros(f3.OutputSize, 1);
f3.BiasL2Factor = 0;
f3.BiasLearnRateFactor = 0;

lgraph = addLayers(lgraph, f3);
lgraph = connectLayers(lgraph, 'relu2', 'fc3');

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

f5 = fullyConnectedLayer(numOfClasses, 'Name', 'pred');
f5.Bias = zeros(f5.OutputSize, 1);
f5.BiasL2Factor = 0;
f5.BiasLearnRateFactor = 0;
f5.Weights = ECOC;
f5.WeightL2Factor = 0 ;
f5.WeightLearnRateFactor = 0;

lgraph = addLayers(lgraph, f5);
lgraph = connectLayers(lgraph, 'tanh', 'pred');

lgraph = addLayers(lgraph, depthConcatenationLayer(2, 'Name', 'concat_clssf_clss'));
lgraph = connectLayers(lgraph, 'tanh', 'concat_clssf_clss/in1');
lgraph = connectLayers(lgraph, 'pred', 'concat_clssf_clss/in2');

lgraph = addLayers(lgraph, MSE_Comb('loss', ECOC, 0.5));
lgraph = connectLayers(lgraph, 'concat_clssf_clss', 'loss');

options = trainingOptions('rmsprop', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 256, ...
    'Verbose',true);
tic;
[net, ~] = trainNetwork(TrainFeatures, GT_EcocMapped, lgraph, options);
tim_Train = toc';

end


