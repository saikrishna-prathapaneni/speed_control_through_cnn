%%
%imgSetVector = imageSet('myData.mat','recursive');
labels = readtable('labels.csv');
clc;
%analyzeNetwork(resnet50());
%Initializing arduino

%%
file=fullfile('mydata');

cat={'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'...
    ,'16','17','18','19','20','21','22','23','24','25','26','27','28'...
    '29','30','31','32','33','34','35','36','37','38','39','40','41','42'};

%labels = readtable('labels.csv');
data_set=imageDatastore(fullfile(file,cat),'LabelSource','foldernames');
count=countEachLabel(data_set);
min_value=min(count{:,2});

data_set=splitEachLabel(data_set,min_value,'randomize');
countEachLabel(data_set);

%to display the image

%sign_20 =  find(data_set.Labels == '10' , 3);
%sign_30 =  find(data_set.Labels == '1' , 3);
%sign_40 =  find(data_set.Labels == '2' , 3);

%m=length(sign_20);
%for i=1:m
  %  subplot(3,3,i);
 %   imshow(readimage(data_set,sign_20(i)));
%end


%to know the structure of the cnn
%%
     
net = resnet50
net = resnet50('Weights','imagenet')
lgraph = resnet50('Weights','none')
%plot(net);
%set(gca,'YLim',[150 170]);

%know the no. of nodes in input layer and number of nodes in the output
%layer
analyzeNetwork(net)
net.Layers(1);
output_number=numel(net.Layers(end).ClassNames);
inputsize_of_cnn=net.Layers(1).InputSize;

%%

% a coustem layer

layers = [
    imageInputLayer(inputsize_of_cnn,'Name','input')
    
    convolution2dLayer(5,16,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')
    
    additionLayer(2,'Name','add')
    
    averagePooling2dLayer(2,'Stride',2,'Name','avpool')
    fullyConnectedLayer(42,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];
lgraph_3=layerGraph;
lgraph_3 = layerGraph(layers);

skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
lgraph_3 = addLayers(lgraph_3,skipConv);


lgraph_3 = connectLayers(lgraph_3,'relu_1','skipConv');
lgraph_3 = connectLayers(lgraph_3,'skipConv','add/in2');







%%
%%
%dividing the data into  training and test data

[train_data,test_data]=splitEachLabel(data_set,0.8,'randomize');

%changing the input data size to required size

augumented_train_data=augmentedImageDatastore(inputsize_of_cnn,train_data);
augumented_test_data=augmentedImageDatastore(inputsize_of_cnn,test_data);

%know the weights of the 1st layer 
weight_1=net.Layers(2).Weights;
weight_1=mat2gray(weight_1);
%figure
%montage(weight_1);
%%
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
%%
%obtaining features from the one of the layers of neural network
featurelayer = 'fc1000';
training_features=activations(net,augumented_train_data...
    ,featurelayer,'MiniBatchSize',30,'OutputAs','columns');
training_features=transpose(training_features);
%obtain labels from the train_data
traininglables=train_data.Labels;
testlables=test_data.Labels;
%%
%create the classifier for the present training features
classifier=fitcecoc(training_features,traininglables...
    ,'Learner','Linear','Coding','onevsall');

%time for the prediction(how well the data model is performed over the test data)
test_features=activations(net,augumented_test_data...
    ,featurelayer,'MiniBatchSize',32,'OutputAs','columns');     
test_features=transpose(test_features);
%%
predict_labels=predict(classifier,test_features);    

testlables=test_data.Labels;
confmat=confusionmat(testlables,predict_labels);


%know the percentage of the accuracy of the trained data
k=bsxfun(@rdivide,confmat,sum(confmat,2));
mean(diag(k))
%percentage of accuracy of the given classifier is around 84 percentage
%%
options = trainingOptions('sgdm', ...
    'MaxEpochs',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{augumented_test_data,testlables}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%%
net_1 = trainNetwork(augumented_train_data,traininglables,lgraph_3,options);

plotconfusion(testlables,predict_labels)
%%
%test the other data_set for the classification

img = imread(fullfile('test_data_5.jpg'));
 
 test=augmentedImageDatastore(inputsize_of_cnn,img,'ColorPreprocessing','gray2rgb');
 testk=activations(net,test...
     ,featurelayer,'MiniBatchSize',32,'OutputAs','columns');     
 testk=transpose(testk);
 label=predict(classifier,testk);
 label
 %k=int16(label)
 %k=k+6.5;
 %k=int16(k)
%labels.(2)(k)
%%
ard_obj = arduino('COM9','Nano','Libraries','Adafruit\MotorShield')
Shield = addon(ard_obj,'Adafruit\MotorShieldV2');
dc_motor=dcmotor(Shield,2);
servo_m= servo(Shield,1);
%%
if label == '13'
    dc_motor.Speed(0.6);
    start(dc_motor);
    pause(5);
    dc_motor.speed(0.5);
    stop(dc_motor);   
end

if label == '33' % turn right
    for count=1:5
        for angle = 0:0.1:0.5
            writePosition(servo_m,angle)  %From 0 to 90 degrees for Steering 
        end
        pause();% shall be based on other parameters
        for angle = 0.5:-1:0
            writePosition(servo_m,angle) % setting the steering position to 0
        end
    end
end


 if label == '4'
     fprintf("dc activated")
    dc_motor.Speed(0.7);
    start(dc_motor);
    pause(5);
    dc_motor.speed(0.5);
    stop(dc_motor);
 end 
 
 if label == '40' % label for roundabout
     fprintf("servo activated")
      for count=1:5
        for angle = 0:0.1:1
            writePosition(servo_m,angle)  %From 0 to 180 degrees for Steering 
        end
        pause();% shall be based on other parameters
        for angle = 1:-1:0
            writePosition(servo_m,angle) % setting the steering position to 0
        end
      end
 end     
        
        
 
%%