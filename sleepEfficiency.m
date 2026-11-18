clear all; clc; close all;
rng(1);

% loading the dataset
sleepFile = readtable("Sleep_Efficiency.csv");
chosenCols = ["age","sleepDuration","remSleepPercentage","deepSleepPercentage","awakenings","caffeineConsumption","alcoholConsumption","exerciseFrequency"];

x = sleepFile{:, chosenCols};
y = categorical(sleepFile.sleepEfficiency >= 0.8);

% Here I split the dataset into train and test sets with a 70/30 split
split = cvpartition(size(sleepFile, 1), 'HoldOut', 0.3);
xTrain = x(training(split), :);
yTrain = y(training(split), :);
xTest = x(test(split), :);
yTest = y(test(split), :);

% logistic model test accuracy when run once
logistic = fitglm(xTrain, yTrain, "distribution", "binomial", "link", "logit");
logisticA = predict(logistic, xTest);
logisticB = logisticA >= 0.5;
logisticB = categorical(logisticB);
logisticAccuracy = sum(logisticB == yTest) / length(yTest);
fprintf('logistic accuracy after just one run: %.2f%%\n', logisticAccuracy * 100);

% random forest model test accuracy when run once
randomForest = TreeBagger(300, xTrain, yTrain, 'Method', 'classification', 'OOBPrediction','on', 'OOBPredictorImportance','on');
forestA = predict(randomForest, xTest);
forestA = categorical(forestA);
forestAccuracy = sum(forestA == yTest) / length(yTest);
fprintf('random forest accuracy after just one run: %.2f%%\n', forestAccuracy * 100);

% finding the training accuracy of my logistic model
logisticTrain = predict(logistic, xTrain);
logisticTrain = logisticTrain >= 0.5;
logisticTrain = categorical(logisticTrain);
logisticTrainAccuracy = sum(logisticTrain == yTrain) / length(yTrain);
fprintf('\nlogistic training accuracy: %.2f%%\n', logisticTrainAccuracy * 100);

% finding the training accuracy of my random forest model
forestTrain = predict(randomForest, xTrain);
forestTrain = categorical(forestTrain);
forestTrainAccuracy = sum(forestTrain == yTrain) / length(yTrain);
fprintf('random forest training accuracy: %.2f%%\n', forestTrainAccuracy * 100);

% k fold cross validation for both model
k = 10;
kSplit = cvpartition(yTrain, "KFold", k);
kLogAccuracy = [];
kForestAccuracy = [];

label = [];
logScore = [];
forScore = [];

kLogTime = [];
kForestTime = [];

for i = 1:k
    xkTrain = xTrain(training(kSplit, i), :);
    ykTrain = yTrain(training(kSplit, i), :);
    xkTest = xTrain(test(kSplit, i), :);
    ykTest = yTrain(test(kSplit, i), :);

    tic
    kLogistic = fitglm(xkTrain, ykTrain, "distribution", "binomial", "link", "logit");
    kLogTime(end+1) = toc;
    logA = predict(kLogistic, xkTest);
    logB = logA >= 0.5;
    logB = categorical(logB);
    kLogAccuracy(end+1) = sum(logB == ykTest) / length(ykTest);

    tic
    kRandomForest = TreeBagger(300, xkTrain, ykTrain, 'Method', 'classification', 'OOBPrediction','on', 'OOBPredictorImportance','on');
    kForestTime(end+1) = toc;
    forA = predict(kRandomForest, xkTest);
    forA = categorical(forA);
    [~, forB] = predict(kRandomForest, xkTest);
    kForestAccuracy(end+1) = sum(forA == ykTest) / length(ykTest);

    % Storing the data that im going to use for the ROC graph
    logScore = [logScore; logA];
    forScore = [forScore; forB(:, 2)]; 
    label = [label; double(ykTest == categorical(true))];
end

% outputting my k fold accuracy for each model
fprintf('\nThe 10 fold logistic accuracy is: %.2f%%\n', mean(kLogAccuracy)*100);
fprintf('The 10 fold random forest accuracy is: %.2f%%\n', mean(kForestAccuracy)*100);

% outputting my training time of each model
fprintf('\nAverage logistic training time is: %.4f seconds\n', mean(kLogTime));
fprintf('Average random forest training time is: %.4f seconds\n', mean(kForestTime));

% making a heatmap for columns in sleep efficiency file
num = varfun(@isnumeric, sleepFile, 'OutputFormat','uniform');
numTable = sleepFile(:, num);
vars = numTable.Properties.VariableNames;
X = table2array(numTable);
R = corr(X, 'rows', 'pairwise');

figure;
h = heatmap(vars, vars, R);
h.Title = 'Correlation Matrix';
h.CellLabelFormat = '%.2f';
colormap(winter)
colorbar

% logistic model confusion matrix
figure;
logCm = confusionchart(yTest, logisticB);
logCm.Title = 'logistic confusion matrix';

% Display the confusion matrix for the random forest model
figure;
forestCm = confusionchart(yTest, forestA);
forestCm.Title = 'random forest confusion matrix';

% Evaluate the feature importance from the random forest model
importance = randomForest.OOBPermutedPredictorDeltaError;
figure;
bar(importance);
xlabel('Features');
ylabel('Importance');
title('Feature Importance from my Random Forest Model');
set(gca, 'XTickLabel', vars, 'XTickLabelRotation', 45);

% boxplots for chosenCols vs sleep efficiency
figure;
for i = 1:length(chosenCols)
    subplot(3, 3, i);
    boxplot(sleepFile{:, chosenCols(i)}, y, 'Labels', {'Inefficient', 'Efficient'}, 'Colors', 'rg');
    ylabel(chosenCols(i));
    title(chosenCols(i) + ' vs Sleep Efficiency');
end

% pie chart for number of efficient vs inefficient sleepers
efficientTotal = sum(y == 'true');
inefficientTotal = sum(y == 'false');

figure;
pie([efficientTotal, inefficientTotal], {'Efficient', 'Inefficient'});
title('Distribution of Sleep Efficiency Pie Chart');

% bar chart for number of efficient vs inefficient sleepers
figure;
bar([efficientTotal, inefficientTotal]);
set(gca, 'XTickLabel', {'Efficient', 'Inefficient'});
ylabel('Number of People');
title('Bar chart of sleep efficiency distribution');

% making a ROC curve for logistic k fold vs random forest k fold
[xLogistic, yLogistic, ~, logisticAuc] = perfcurve(label, logScore, 1);
[xForest, yForest, ~, forestAuc] = perfcurve(label, forScore, 1);

figure;
plot(xLogistic, yLogistic, 'r', 'LineWidth', 1.5);
hold on;
plot(xForest, yForest, 'b', 'LineWidth', 1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Logistic vs Random Forest');
hold off;

% outputting AUC values for both of the models
fprintf('\nLogistic AUC: %.4f\n', logisticAuc);
fprintf('Random Forest AUC: %.4f\n', forestAuc);