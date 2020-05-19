%imgSetVector = imageSet('myData.mat','recursive');
%labels = readtable('labels.csv');
clc;
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

     
net=resnet50();
%plot(net);
%set(gca,'YLim',[150 170]);

%know the no. of nodes in input layer and number of nodes in the output
%layer

net.Layers(1);
output_number=numel(net.Layers(end).ClassNames);
inputsize_of_cnn=net.Layers(1).InputSize;

%dividing the data into validation and training and test data

[train_data,test_data]=splitEachLabel(data_set,0.8,'randomize');

%changing the input data size to required size

augumented_train_data=augmentedImageDatastore(inputsize_of_cnn,train_data);
augumented_test_data=augmentedImageDatastore(inputsize_of_cnn,test_data);

%know the weights of the 1st layer 
weight_1=net.Layers(2).Weights;
weight_1=mat2gray(weight_1);
%figure
%montage(weight_1);


%obtaining features from the one of the layers of neural network
featurelayer = 'fc1000';
training_features=activations(net,augumented_train_data...
    ,featurelayer,'MiniBatchSize',30,'OutputAs','columns');
training_features=transpose(training_features);
%obtain labels from the train_data
traininglables=train_data.Labels;

%create the classifier for the present training features
classifier=fitcecoc(training_features,traininglables...
    ,'Learner','Linear','Coding','onevsall');

%time for the prediction(how well the data model is performed over the test data)
test_features=activations(net,augumented_test_data...
    ,featurelayer,'MiniBatchSize',32,'OutputAs','columns');     
test_features=transpose(test_features);
predict_labels=predict(classifier,test_features);    

testlables=test_data.Labels;
confmat=confusionmat(testlables,predict_labels);


%know the percentage of the accuracy of the trained data
k=bsxfun(@rdivide,confmat,sum(confmat,2));
mean(diag(k))
%percentage of accuracy of the given classifier is arpund 77 percentage


%test the other data_set for the classification

img = imread(fullfile('test_data_4.jpg'));
 
 test=augmentedImageDatastore(inputsize_of_cnn,img,'ColorPreprocessing','gray2rgb');
 testk=activations(net,test...
     ,featurelayer,'MiniBatchSize',32,'OutputAs','columns');     
 testk=transpose(testk);
 label=predict(classifier,testk);
 label
 k=int16(label)
 k=k+6.5;
 k=int16(k)
labels.(2)(k)

% fprintf('label is %s',label);

