
% add vlfeat required paths
setup ;

% load training data
positive = load('data/aeroplane_train_hist.mat') ;
negative = load('data/background_train_hist.mat') ;
trainNames = {positive.names{:}, negative.names{:}};
trainHistograms = [positive.histograms, negative.histograms] ;
trainLabels = [ones(1,numel(positive.names)), - ones(1,numel(negative.names))] ;
clear positive negative ;

% load testing data
positive = load('data/aeroplane_val_hist.mat') ;
negative = load('data/background_val_hist.mat') ;
testNames = {positive.names{:}, negative.names{:}};
testHistograms = [positive.histograms, negative.histograms] ;
testLabels = [ones(1,numel(positive.names)), - ones(1,numel(negative.names))] ;
clear positive negative ;

fraction = +inf ;

sel = vl_colsubset(1:numel(trainLabels), fraction, 'uniform') ;
trainNames = trainNames(sel) ;
trainHistograms = trainHistograms(:,sel) ;
trainLabels = trainLabels(:,sel) ;
clear sel ;

% display number of training and testing samples
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(trainLabels > 0), sum(trainLabels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% normalize the histograms before running the linear SVM
trainHistograms = sqrt(trainHistograms) ;
testHistograms = sqrt(testHistograms) ;

% train the linear SVM.
[w, bias] = trainLinearSVM(trainHistograms, trainLabels, 60) ;

% evaluate the scores on the training data
trainScores = w' * trainHistograms + bias ;

% display the ranked list of images
figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
displayRankedImageList(trainNames, trainScores)  ;

% test the linar SVM
testScores = w' * testHistograms + bias ;

% display the ranked list of images
figure(2) ; clf ; set(2,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ;

% print results
[drop,perm] = sort(testScores,'descend') ;
fprintf('Real positive samples in the top 64 images: %d\n', sum(testLabels(perm(1:64)) > 0)) ;
