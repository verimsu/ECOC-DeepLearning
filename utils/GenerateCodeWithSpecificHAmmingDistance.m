% generate random code with specific hamming distance

hammDis = 16;
numOfClassifiers = 30;
numOfClasses = 8;

ECOC = zeros(numOfClasses, numOfClassifiers);
for i = 1 : numOfClasses
    
    cd = double(rand(numOfClassifiers, 1) > 0.5);
    cd = cd(randperm(length(cd)));
    while true
        
        sumTr = 0;
        for j = 1 : i - 1
            sumTr = sumTr + double( sum(ECOC(j, :) ~= cd') >= hammDis );
        end
        
        if sumTr == (i-1)
            ECOC(i, :) = cd;
            break
        else
            cd = double(rand(numOfClassifiers, 1) > 0.5);
            cd = cd(randperm(length(cd)));
        end
    end
    
    i
end

save(['ECOC_', int2str(numOfClasses), '_', int2str(numOfClassifiers), '_', int2str(hammDis), '.mat'], 'ECOC')