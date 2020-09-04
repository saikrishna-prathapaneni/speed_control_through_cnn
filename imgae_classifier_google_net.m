labels = readtable('labels.csv');



file=fullfile('mydata');

cat={'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'...
    ,'16','17','18','19','20','21','22','23','24','25','26','27','28'...
    '29','30','31','32','33','34','35','36','37','38','39','40','41','42'};



data_set=imageDatastore(fullfile(file,cat),'LabelSource','foldernames');
count=countEachLabel(data_set);
min_value=min(count{:,2});

data_set=splitEachLabel(data_set,min_value,'randomize');
countEachLabel(data_set);

%%

net = googlenet
inputSize = net.Layers(1).InputSize;

output_number=numel(net.Layers(end).ClassNames);



%%

output_number=numel(net.Layers(end).ClassNames);
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 



%%

learnableLayer = 'loss3-classifier'
 classLayer='output'

%%
[train_data,test_data]=splitEachLabel(data_set,0.8,'randomize');

%changing the input data size to required size

augumented_train_data=augmentedImageDatastore(inputSize,train_data);
augumented_test_data=augmentedImageDatastore(inputSize,test_data);
traininglables=train_data.Labels;
testlables=test_data.Labels;


%%
numClasses = numel(categories(train_data.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    


lgraph = replaceLayer(lgraph,learnableLayer,newLearnableLayer);


%%

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer,newClassLayer);


%%
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])


%%

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%%
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),test_data);

%%
miniBatchSize = 32;
%%valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%
net = trainNetwork(augimdsTrain,lgraph,options);

%%

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == testlables)

%%
idx = randperm(numel(test_data.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(test_data,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%%