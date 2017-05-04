clear
clc

file = fopen('dataset_double_2_5k.txt','w');
dataset = load('dataset_double_2_5k.mat');
dataset = dataset.dataset;
for i=1:length(dataset)
    fwrite(file, sprintf('%f', dataset(i,1)));
    for j=2:43
        fwrite(file, sprintf(',%f', dataset(i,j)));
    end
    fwrite(file, sprintf('\r\n'));
end
