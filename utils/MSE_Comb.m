classdef MSE_Comb < nnet.layer.RegressionLayer
    % custom regression layer with Cross Entropy loss.
    properties
        ECOC
        weightedLoss
    end
    methods
        function layer = MSE_Comb_FixedBackProp(name, ECOC, weightedLoss)
            % layer = CERegressionLayer(name) creates a
            % Cross Entropy regression layer and specifies the layer
            % name.
            
            % Set layer name.
            if nargin >= 1
                % Set layer name.
                layer.Name = name;
            end
            layer.ECOC = ECOC;
            layer.weightedLoss = weightedLoss;
            
            % Set layer description.
            layer.Description = 'Cross Entropy Error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Cross Entropy loss between
            % the predictions Y and the training targets T.
            w = layer.weightedLoss;
            
            [numClasses, numClassifiers] = size(layer.ECOC);

            Y = squeeze(Y);
            T = squeeze(T);
            sY = size( Y );
            numElems = sY(end);
            
            Y_Classifiers = Y(1:numClassifiers, :); T_Classifiers = T(1:numClassifiers, :);
            Classifiers_Loss = 0.5*(Y_Classifiers-T_Classifiers).^2;
            Classifiers_Loss = sum(Classifiers_Loss(:))./numElems;
                        
            Y_Classes = Y(numClassifiers+1:end, :); T_Classes = T(numClassifiers+1:end, :);
            Classes_Loss = 0.5*T_Classes.*(Y_Classes - T_Classes*numClassifiers).^2;
            Classes_Loss = sum(Classes_Loss(:))./numElems;
            
            loss = (1-w)*Classifiers_Loss + w*Classes_Loss;
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the Cross Entropy loss with respect to the predictions Y
            w = layer.weightedLoss;
            
            [numClasses, numClassifiers] = size(layer.ECOC);
            
            szY = size( Y );
            Y = squeeze(Y);
            T = squeeze(T);
            
            numElems = szY(end);
            
            Y_Classifiers = Y(1:numClassifiers, :); T_Classifiers = T(1:numClassifiers, :);
            dLdY_Classifiers = 2*(1-w)*((Y_Classifiers-T_Classifiers))./numElems;
                        
            Y_Classes = Y(numClassifiers+1:end, :); T_Classes = T(numClassifiers+1:end, :);
            dLdY_Classes = 2*(w)*T_Classes.*((Y_Classes - T_Classes*numClassifiers))./numElems;
            
            
            dLdY = cat(1, dLdY_Classifiers, dLdY_Classes);
            dLdY = reshape(dLdY, szY);
        end
        
    end
end