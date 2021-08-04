function layer = fullyConnectedLayer_SubSplit( varargin )
% fullyConnectedLayer   Fully connected layer
%
%   layer = fullyConnectedLayer(outputSize) creates a fully connected
%   layer. outputSize specifies the size of the output for the layer. A
%   fully connected layer will multiply the input by a matrix and then add
%   a bias vector.
%
%   layer = fullyConnectedLayer(outputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Weights'                 - Layer weights, specified as an 
%                                   outputSize-by-inputSize matrix or [],
%                                   where inputSize is the input size of
%                                   the layer. The default is [].
%       'Bias'                    - Layer biases, specified as an
%                                   outputSize-by-1 matrix or []. The
%                                   default is [].
%       'WeightLearnRateFactor'   - A number that specifies multiplier for
%                                   the learning rate of the weights. The
%                                   default is 1.
%       'BiasLearnRateFactor'     - A number that specifies a multiplier
%                                   for the learning rate for the biases.
%                                   The default is 1.
%       'WeightL2Factor'          - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   weights. The default is 1.
%       'BiasL2Factor'            - A number that specifies a multiplier
%                                   for the L2 weight regulariser for the
%                                   biases. The default is 0.
%       'WeightsInitializer'      - The function to initialize the weights,
%                                   specified as 'glorot', 'he',
%                                   'orthogonal', 'narrow-normal', 'zeros',
%                                   'ones' or a function handle. The
%                                   default is 'glorot'.
%       'BiasInitializer'         - The function to initialize the bias,
%                                   specified as 'narrow-normal', 'zeros', 'ones' 
%                                   or a function handle. The default is 
%                                   'zeros'.
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%   Example 1:
%       Create a fully connected layer with an output size of 10, and an
%       input size that will be determined at training time.
%
%       layer = fullyConnectedLayer(10);
%
%   See also nnet.cnn.layer.FullyConnectedLayer, convolution2dLayer,
%   reluLayer.

%   Copyright 2015-2018 The MathWorks, Inc.

% Parse the input arguments.
args = iParseInputArguments(varargin{:});

% Create an internal representation of a fully connected layer.
internalLayer = FullyConnected_SubSplit( ...
    args.Name, ...
    args.InputSize, ...
    args.OutputSize, ...
    args.SplitN);

internalLayer.Weights.L2Factor = args.WeightL2Factor;
internalLayer.Weights.LearnRateFactor = args.WeightLearnRateFactor;

internalLayer.Bias.L2Factor = args.BiasL2Factor;
internalLayer.Bias.LearnRateFactor = args.BiasLearnRateFactor;

% Pass the internal layer to a function to construct a user visible
% fully connected layer.
layer = nnet.cnn.layer.FullyConnectedLayer(internalLayer);
layer.WeightsInitializer = args.WeightsInitializer;
layer.BiasInitializer = args.BiasInitializer;
layer.Weights = args.Weights;
layer.Bias = args.Bias;
end

function inputArguments = iParseInputArguments(varargin)
varargin = nnet.internal.cnn.layer.util.gatherParametersToCPU(varargin);
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultWeightLearnRateFactor = 1;
defaultBiasLearnRateFactor = 1;
defaultWeightL2Factor = 1;
defaultBiasL2Factor = 0;
defaultWeightsInitializer = 'glorot';
defaultBiasInitializer = 'zeros';
defaultName = '';
defaultLearnable = [];
defaultSplitN = 1;

p.addRequired('OutputSize', @iAssertValidOutputSize);
p.addParameter('WeightLearnRateFactor', defaultWeightLearnRateFactor, @iAssertValidFactor);
p.addParameter('BiasLearnRateFactor', defaultBiasLearnRateFactor, @iAssertValidFactor);
p.addParameter('WeightL2Factor', defaultWeightL2Factor, @iAssertValidFactor);
p.addParameter('BiasL2Factor', defaultBiasL2Factor, @iAssertValidFactor);
p.addParameter('WeightsInitializer', defaultWeightsInitializer);
p.addParameter('BiasInitializer', defaultBiasInitializer);
p.addParameter('Name', defaultName, @iAssertValidLayerName);
p.addParameter('Weights', defaultLearnable);
p.addParameter('Bias', defaultLearnable);
p.addParameter('SplitN', defaultSplitN);
end

function iAssertValidFactor(value)
nnet.internal.cnn.layer.paramvalidation.validateLearnFactor(value);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function inputArguments = iConvertToCanonicalForm(p)
% Make sure integral values are converted to double and strings to char vectors
inputArguments = struct;
inputArguments.InputSize = [];
inputArguments.OutputSize = double( p.Results.OutputSize );
inputArguments.WeightLearnRateFactor = p.Results.WeightLearnRateFactor;
inputArguments.BiasLearnRateFactor = p.Results.BiasLearnRateFactor;
inputArguments.WeightL2Factor = p.Results.WeightL2Factor;
inputArguments.BiasL2Factor = p.Results.BiasL2Factor;
inputArguments.WeightsInitializer = p.Results.WeightsInitializer;
inputArguments.BiasInitializer = p.Results.BiasInitializer;
inputArguments.Name = char(p.Results.Name);
inputArguments.Weights = p.Results.Weights;
inputArguments.Bias = p.Results.Bias;
inputArguments.SplitN = p.Results.SplitN;
end

function iAssertValidOutputSize(value)
validateattributes(value, {'numeric'}, ...
    {'nonempty', 'scalar', 'integer', 'positive'});
end
