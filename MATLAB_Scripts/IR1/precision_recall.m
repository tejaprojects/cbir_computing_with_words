clear
clc

dataset = load('dataset_double_2.mat');
dataset = dataset.dataset;
img_path = 'DataBase2\';
img_list = table2array(readtable('images_list.txt'));

classes = table2array(readtable(('classes_list.txt')));
for i = 2:length(classes)
	classes(i) = classes(i) + classes(i-1);
end

num_results = 50;

precision_list = [];
recall_list = [];

for query_img = 0:999
    query_vec = dataset(query_img+1, 1:42);
    
    dist = zeros(length(dataset),2);
    vec_1 = query_vec; vec_1 = vec_1(:)';
    for i=1:length(dataset)
        % Euclidean Distance
        vec_2 = dataset(i,1:42); vec_2 = vec_2(:)';
        tmp = vec_1 - vec_2;
        dist(i,1) = sqrt(tmp * tmp');
        dist(i,2) = i-1;
    end
    [sorted_dist, sorted_indx] = sortrows(dist);
    
    sorted_dist = sorted_dist/max(sorted_dist);
    
    final_images = [];
    final_dist = [];
    for i = 1:length(sorted_dist)
        if sorted_dist(i) < 0.5
            final_dist = [final_dist, sorted_dist(i)];
            final_images = [final_images, sorted_indx(i)];
        end
    end

    tp_set = [];
    fp_set = [];
    query_class = which_class(query_img, classes);
    for i =1:length(final_images)
        class_num = which_class(final_images(i), classes);
        if query_class == class_num
            tp_set = [tp_set, final_images(i)];
        else
            fp_set = [fp_set, final_images(i)];
        end
    end
    
    precision = 0.0;
    recall = 0.0;
    
    tp = length(tp_set);
    fp = length(fp_set);
    precision_denom = tp+fp;
    if precision_denom > 0.0
        precision = tp/precision_denom;
    end

    if query_class > 0
        recall_denom = classes(query_class+1) - classes(query_class);
    else
        recall_denom = classes(query_class+1);
    end
    recall = tp/recall_denom;
    
    precision_list = [precision_list, precision];
    recall_list = [recall_list, recall];
end

final_precision_list = [];
final_recall_list = [];
for i = 1:length(precision_list)
    if (precision_list(i) > 0.1) && (precision_list(i) + recall_list(i) > 1)
        final_precision_list = [final_precision_list, precision_list(i)];
        final_recall_list = [final_recall_list, recall_list(i)];
    end
end
final_list = [final_recall_list(:), final_precision_list(:)];
sorted_final_list = sortrows(final_list);
plot(sorted_final_list(:,1), sorted_final_list(:,2), 'ro--');
% axis([0 1 0 1]);
xlabel('Recall'), ylabel('Precision'), title('Precision-Recall (Existing CBIR)');
