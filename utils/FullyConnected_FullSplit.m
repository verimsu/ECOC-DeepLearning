classdef FullyConnected_FullSplit < nnet.internal.cnn.layer.FunctionalLayer ...
    & nnet.internal.cnn.layer.CPUFusableLayer
    % FullyConnected   Implementation of the fully connected layer
    
    %   Copyright 2015-2020 The MathWorks, Inc.
    
    properties
        % LearnableParameters   The learnable parameters for this layer
        %   This layer has two learnable parameters, which are the weights
        %   and the bias.
        LearnableParameters = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter.empty()
        
        % Name (char array)   A name for the layer
        Name
        SplitN
    end
    
    properties (Constant)
        % DefaultName   Default layer's name.
        DefaultName = 'fc'
    end
    
    properties(Access = public)
        QuantizationMethod
        
        % ParConverter    Converts internal-external learnable parameters
        ParConverter = ...
            nnet.internal.cnn.layer.util.FullyConnectedParsConverter();
    end
    
    properties(SetAccess = private)
        % InputNames   This layer has a single input
        InputNames = {'in'}
        
        % OutputNames   This layer has a single output
        OutputNames = {'out'}
        
        % InputSize   Input size of the layer
        %   The input size of the fully connected layer. Note that the
        %   internal layers deal with 3D observations, and so this input
        %   size will be 3D. This will be empty if the input size has not
        %   been set yet.
        InputSize
        
        % NumNeurons  (scalar integer)   Number of neurons for the layer
        NumNeurons
        
        % Execution strategy   Execution strategy of the layer
        %   The execution strategy determines where (host/GPU) and how
        %   forward and backward operations are performed.
        ExecutionStrategy
        
        % ObservationDimension for the input data
        ObservationDim
    end
    
    properties (Dependent)
        % Weights   The weights for the layer
        Weights
        
        % Bias   The bias vector for the layer
        Bias
        
        % Learnables   Cell array with dlarrays
        Learnables
    end
    
    properties(SetAccess=protected, GetAccess=?nnet.internal.cnn.dlnetwork)
        LearnablesNames = ["Weights" "Bias"]
    end    
    
    properties (Dependent, SetAccess = private)
        % HasSizeDetermined   Specifies if all size parameters are set
        %   If the input size has not been determined, then this will be
        %   set to false, otherwise it will be true.
        HasSizeDetermined
        
        % Expected external Weights size
        ExtWeightsSize
        
        % Expected external Bias size
        ExtBiasSize
    end
    
    properties (Constant, Access = private)
        % WeightsIndex   Index of the Weights in the LearnableParameter vector
        WeightsIndex = 1;
        
        % BiasIndex   Index of the Bias in the LearnableParameter vector
        BiasIndex = 2;
    end
    
    methods
        function this = FullyConnected_FullSplit(name, inputSize, numNeurons, SplitN)
            this.Name = name;
            
            % Set hyperparameters
            this.NumNeurons = numNeurons;
            this.InputSize = inputSize;
            this.SplitN = SplitN;
            if ~isempty(this.InputSize)
                if isempty(this.ObservationDim)
                    this.ObservationDim = numel(this.InputSize)+1;
                end
            end
            
            % Set learnable parameters
            this.Weights = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            this.Bias = nnet.internal.cnn.layer.learnable.PredictionLearnableParameter();
            % Set default initializers. The external layer constructor
            % overwrites these values, which are set only for internal code
            % that bypasses the casual API.
            this.Weights.Initializer = iInternalInitializer('narrow-normal');
            this.Bias.Initializer = iInternalInitializer('zeros');
            
            % Set execution strategy
            this.ExecutionStrategy = this.getHostStrategy();
            
            % Initialize QuantizationMethod
            this.QuantizationMethod = nnet.internal.cnn.layer.NoQuantization; 

            % FullyConnected layer needs X but not Z for the backward pass
            this.NeedsXForBackward = true;
            this.NeedsZForBackward = false;            
        end
        
        function Z = predict(this, X)
            [wieghts0, bias0] = this.QuantizationMethod.remapped(this.Weights.Value, this.Bias.Value);
            W = wieghts0;
            b = bias0;
            
            
            szX = size(X);
            szW = size(W);
            %Z = reshape( squeeze(W)' * squeeze(X), [1, 1, szW(end), szX(end)] ) + b;
            
            W2 = squeeze(W);
            X2 = squeeze(X);
            b2 = squeeze(b);
            
            Gap1 = szW(3)/this.SplitN;
            Gap2 = szW(4)/this.SplitN;
            Z = gpuArray(zeros(szW(end), szX(end), 'like', W));
            for i = 1 : this.SplitN
                p1 = (i-1)*Gap1 + 1: (i)*Gap1;
                p2 = (i-1)*Gap2 + 1: (i)*Gap2;
                
                Z(p2, : ) = W2(p1, p2)' * X2(p1, :) + b2(p2);
            end
            Z = reshape( Z, [1, 1, szW(end), szX(end)] );
        end
        
        
        function [Z, memory] = forward(this, X)
            %Z = this.ExecutionStrategy.forward(X, this.Weights.Value,this.Bias.Value, this.ObservationDim);
            
            W = this.Weights.Value;
            b = this.Bias.Value;
            szX = size(X);
            szW = size(W);
            
            %Z = reshape( squeeze(W)' * squeeze(X), [1, 1, szW(end), szX(end)] ) + b;
            
            W2 = squeeze(W);
            X2 = squeeze(X);
            b2 = squeeze(b);
            
            Gap1 = szW(3)/this.SplitN;
            Gap2 = szW(4)/this.SplitN;
            Z = gpuArray(zeros(szW(end), szX(end), 'like', W));
            
            for i = 1 : this.SplitN
                p1 = (i-1)*Gap1 + 1: (i)*Gap1;
                p2 = (i-1)*Gap2 + 1: (i)*Gap2;
                
                Z(p2, : ) = W2(p1, p2)' * X2(p1, :) + b2(p2);
            end
            Z = reshape( Z, [1, 1, szW(end), szX(end)] );
            
            
            memory = [];
        end
        
        function varargout = backward(this, X, ~, dZ, ~)
            %[varargout{1:nargout}] = this.ExecutionStrategy.backward(X, this.Weights.Value, dZ, this.ObservationDim);
            W = this.Weights.Value;
            b = this.Bias.Value;
            
            szX = size(X);
            szW = size(W);
            
            W2 = squeeze(W);
            dZ2 = squeeze(dZ);
            X2 = squeeze(X);
            b2 = squeeze(b);
            
            dX = gpuArray(zeros(szX(3:end), 'like', X));
            
            Gap1 = szW(3)/this.SplitN;
            Gap2 = szW(4)/this.SplitN;
            for i = 1 : this.SplitN
                p1 = (i-1)*Gap1 + 1: (i)*Gap1;
                p2 = (i-1)*Gap2 + 1: (i)*Gap2;
                dX(p1, :) = W2(p1, p2)*dZ2(p2, :)+ b2(p2);
            end
            dX = reshape( dX, szX );
            
            dW = gpuArray(zeros(szW(3:end), 'like', W));
            for i = 1 : this.SplitN
                p1 = (i-1)*Gap1 + 1: (i)*Gap1;
                p2 = (i-1)*Gap2 + 1: (i)*Gap2;
                dW(p1, p2) = X2(p1, :)*dZ2(p2, :)';
            end
            dW = reshape( dW, szW );
            
            
            dBias = sum(dZ, length(szX));
            varargout{1} = dX;
            varargout{2}{1} = dW;
            varargout{2}{2} = dBias;
            
        end
       
        
        function this = inferSize(this, inputSize)
            if ~this.HasSizeDetermined
                if isempty(inputSize) || any(inputSize == 0)
                    % The network analyzer sometimes propagates empty input
                    % sizes or input sizes with only zeros, for example
                    % when the previous layer was not able to compute an
                    % output size. Perhaps other layers can deal with this
                    % sort of input. For a fully connected layer, such edge
                    % case sizes don't really make sense and can cause
                    % problems when trying to reshape learnable parameters.
                    % We circumvent these situations by directly throwing
                    % an error here. The network analyzer will catch errors
                    % and consider the input size to be invalid. 
                    error(message('nnet_cnn:internal:cnn:layer:FullyConnected:InvalidInputSize'))
                end
                this.InputSize = inputSize;
                this.ObservationDim = numel(inputSize) + 1;
                % Match weights and bias to layer size
                this.LearnableParameters(this.WeightsIndex).Value = ...
                    this.ParConverter.toInternal(this.Weights.Value,...
                        this.InputSize,this.NumNeurons,...
                        this.ObservationDim, 'Weights');

                this.LearnableParameters(this.BiasIndex).Value = ...
                    this.ParConverter.toInternal(this.Bias.Value,...
                        this.InputSize,this.NumNeurons,...
                        this.ObservationDim, 'Bias');

                % Now that we know the input size, we can set the execution
                % strategy of the layer -- since the strategy depends on
                % the number of spatial dimensions in the input size. For
                % functional behaviour, we have a one-size-fits-all
                % strategy so we do not need to set the strategy here
                if ~this.isFunctional
                    this.ExecutionStrategy = this.getHostStrategy();
                end
            end
        end
        
        function outputSize = forwardPropagateSize(this, inputSize)
            if ~this.HasSizeDetermined
                error(message('nnet_cnn:internal:cnn:layer:FullyConnected:ForwardPropagateSizeWithNoInputSize'));
            else
                if numel(this.InputSize) ~= 1 && ~this.isFunctional
                    % spatialDims does not include channel and ObservationDim
                    spatialDims = 1:this.ObservationDim-2;
                    filterSize = this.InputSize(spatialDims);
                    outputSpatialDims = floor(inputSize(spatialDims) - filterSize) + 1;
                    outputSize = [ outputSpatialDims this.NumNeurons ];
                else
                    outputSize = this.NumNeurons;
                end
            end
        end
        
        function tf = isValidInputSize(this, inputSize)
            % isValidInputSize   Check if the layer can accept an input of
            % a certain size
            tf = false;
            if ~isempty(inputSize) && all(inputSize > 0)
                if ~this.HasSizeDetermined
                    tf = (numel(inputSize) == 4 || numel(inputSize) == 3 || numel(inputSize) == 1);
                elseif this.isFunctional
                    tf = isequal(prod(this.InputSize), prod(inputSize));
                else
                    tf = isequal(this.InputSize, inputSize);
                end
            end
        end
        
        function outputSeqLen = forwardPropagateSequenceLength(this, inputSeqLen, ~)
            % forwardPropagateSequenceLength   The sequence length of the
            % output of the layer given an input sequence length
            
            if ~isscalar( this.InputSize )
                % For non-scalar input sizes, the layer does not
                % support time distribution
                assert( isnumeric(inputSeqLen{:}) && (inputSeqLen{:} == 1) );
            end
            % A fully connected layer with scalar input size is
            % time-distribtued, and can propagate an arbitrary sequence
            % length.
            outputSeqLen = inputSeqLen;
        end
        
        function this = initializeLearnableParameters(this, precision)
            % Initialize weights
            if isempty(this.Weights.Value)
                % Initialize only if it is empty
                % Initialize using the user visible size
                userVisibleSize = [this.NumNeurons prod(this.InputSize)];
                weights = this.Weights.Initializer.initialize(...
                    userVisibleSize, 'Weights');
                % No need to reshape here since done in the set method,
                % like when setting Weights from external layer
                this.Weights.Value = precision.cast(weights);
            else
                % Cast to desired precision
                this.Weights.Value = precision.cast(this.Weights.Value);
            end
            
            % Initialize bias
            if isempty(this.Bias.Value)
                % Initialize only if it is empty
                % Initialize using the user visible size
                userVisibleSize = [this.NumNeurons 1];
                bias = this.Bias.Initializer.initialize(...
                    userVisibleSize, 'Bias');
                % No need to reshape here since done in the set method,
                % like when setting Bias from external layer
                this.Bias.Value = precision.cast(bias);
            else
                % Cast to desired precision
                this.Bias.Value = precision.cast(this.Bias.Value);
            end
        end
        
        function this = prepareForTraining(this)
            % prepareForTraining   Prepare this layer for training
            %   Before this layer can be used for training, we need to
            %   convert the learnable parameters to use the class
            %   TrainingLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
        end
        
        function this = prepareForPrediction(this)
            % prepareForPrediction   Prepare this layer for prediction
            %   Before this layer can be used for prediction, we need to
            %   convert the learnable parameters to use the class
            %   PredictionLearnableParameter.
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2prediction(this.LearnableParameters);
        end
        
        function this = setupForHostPrediction(this)
            this.ExecutionStrategy = this.getHostStrategy();
            this.LearnableParameters(1).UseGPU = false;
            this.LearnableParameters(2).UseGPU = false;
        end
        
        function this = setupForGPUPrediction(this)
            this.ExecutionStrategy = this.getGPUStrategy();
            this.LearnableParameters(1).UseGPU = true;
            this.LearnableParameters(2).UseGPU = true;
        end
        
        function this = setupForHostTraining(this)
            this.ExecutionStrategy = this.getHostStrategy();
        end
        
        function this = setupForGPUTraining(this)
            this.ExecutionStrategy = this.getGPUStrategy();
        end
        
        function weights = get.Weights(this)
            weights = this.LearnableParameters(this.WeightsIndex);
        end
        
        function this = set.Weights(this, weights)
            if this.HasSizeDetermined && ~isFunctional(this)
                weights.Value = this.ParConverter.toInternal(...
                    weights.Value,this.InputSize,this.NumNeurons,...
                    this.ObservationDim, 'Weights');
            end
            this.LearnableParameters(this.WeightsIndex) = weights;
        end
        
        function bias = get.Bias(this)
            bias = this.LearnableParameters(this.BiasIndex);
        end
        
        function this = set.Bias(this, bias)
            if this.HasSizeDetermined && ~isFunctional(this)
                bias.Value = this.ParConverter.toInternal(...
                    bias.Value,this.InputSize,this.NumNeurons,...
                    this.ObservationDim, 'Bias');

            end
            this.LearnableParameters(this.BiasIndex) = bias;
        end
        
        function learnables = get.Learnables(this)
            % Assume setupForFunctional has been called            
            w = this.Weights.Value;
            b = this.Bias.Value;                
            learnables = {w, b};
        end
        
        function this = set.Learnables(this, learnables)
            % Assume setupForFunctional has been called and
            % HasSizeDetermined
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{1}, this.ExtWeightsSize);
            nnet.internal.cnn.layer.paramvalidation.assertValidLearnables(learnables{2}, this.ExtBiasSize);
            
            this.LearnableParameters(this.WeightsIndex).Value = learnables{1};
            this.LearnableParameters(this.BiasIndex).Value = learnables{2};
        end

        function sz = get.ExtWeightsSize(this)
            if ~isempty(this.InputSize)
                expectedInputSize = prod(this.InputSize);
            else
                expectedInputSize = NaN;
            end
            sz = [this.NumNeurons expectedInputSize];
        end
        
        function sz = get.ExtBiasSize(this)
            sz = [this.NumNeurons 1];
        end
        
        function tf = get.HasSizeDetermined(this)
            tf = ~isempty( this.InputSize );
        end    
    end
    
    methods (Access = private)
        function executionStrategy = getHostStrategy(this)
            switch numel(this.InputSize)
                case 1
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostVectorStrategy();
                case 3
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostImageStrategy();
                case 4
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHost3DImageStrategy();
                otherwise
                   executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedHostImageStrategy(); 
            end
        end
        
        function executionStrategy = getGPUStrategy(this)
            switch numel(this.InputSize)
                case 1
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUVectorStrategy();
                case 3
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUImageStrategy();
                case 4
                    executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPU3DImageStrategy();
                otherwise
                   executionStrategy = nnet.internal.cnn.layer.util.FullyConnectedGPUImageStrategy(); 
            end
        end
        
        function tf = isFunctional(this)
            tf = isa(this.ExecutionStrategy, ...
                'nnet.internal.cnn.layer.util.FunctionalStrategy');
        end
    end
   
    % Overloaded methods to deal with different shape of the internal
    % layers LearnableParameters in functional mode.
    methods
        function this = setupForFunctional(this)
            this.LearnableParameters = nnet.internal.cnn.layer.learnable.convert2training(this.LearnableParameters);
            this = setFunctionalStrategy(this);
            
            if this.HasSizeDetermined && ~(isempty(this.Weights.Value) || isempty(this.Bias.Value))
                % Use external, user visible size. Assume HasSizeDetermined.
                % Weights: [NumNeurons prod(InputSize)]
                % Bias: [NumNeurons 1]
                % This does the same thing that happens when one gets the
                % Weights and Biases from the external layer.
                this.LearnableParameters(this.WeightsIndex).Value = ...
                    this.ParConverter.toExternal(...
                    this.Weights.Value,this.NumNeurons,'Weights');
                this.LearnableParameters(this.BiasIndex).Value = ...
                    this.ParConverter.toExternal(...
                    this.Bias.Value,this.NumNeurons,'Bias');
                
                for i=1:numel(this.LearnableParameters)
                    this.LearnableParameters(i).Value = ...
                        single( dlarray( this.LearnableParameters(i).Value ) );
                end
            end
        end

        function this = revertSetupForFunctional(this)
            this = revertSetupForFunctional@nnet.internal.cnn.layer.FunctionalLayer(this);
                        
            % Use internal size. Assume HasSizeDetermined.
            % The shapes are complicated - see toInternal in
            % nnet.internal.cnn.layer.util.FullyConnectedParsConverter
            % This does the same thing that happens when one sets the
            % Weights and Biases from the external layer. The conversion
            % happens in the set method of Weights and Bias.
            this.Weights.Value = this.Weights.Value;
            this.Bias.Value = this.Bias.Value;
            
            % Now we revert the input size to empty. This is to avoid
            % situations when a layer is moved into a DAGNetwork, where
            % its valid input size will now be of the form [1 1 C].
            this.InputSize = [];
        end    
    end
        
    methods(Access=protected)
        function this = setFunctionalStrategy(this)
            this.ExecutionStrategy = ...
                nnet.internal.cnn.layer.util.FullyConnectedFunctionalStrategy();
        end
    end
        
    methods (Hidden)
        function layerArgs = getFusedArguments(layer)
            % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            W = layer.Weights.Value;
            W = reshape(W, [], size(W, layer.ObservationDim)); 
            if numel(layer.InputSize) == 1
                % Vector strategy has transposed weights.
                W = W';
            end
            layerArgs = { 'fullyconnected', W,...
                squeeze(layer.Bias.Value), layer.InputSize };
        end

        function tf = isFusable(layer, precision, numDataDimensions)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = (numDataDimensions == (length(layer.InputSize)-1)) && ...
                (class(layer.Weights.Value) == precision);
        end
    end
end

function initializer = iInternalInitializer(name)
initializer = nnet.internal.cnn.layer.learnable.initializer...
    .initializerFactory(name);
end

% LocalWords:  Learnable learnable fc hyperparameters nnet cnn Rnd
