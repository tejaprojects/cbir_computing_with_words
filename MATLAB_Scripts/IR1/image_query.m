function [dist] = image_query(query_img)
    dataset = load('dataset_double_2.mat');

    query_vec = dataset(query_img+1);

    dist = zeros(length(dataset),2);
    for i=1:length(dataset)
        % Euclidean Distance
        vec_1 = query_vec; vec_1 = vec_1(:)';
        vec_2 = dataset(i); vec_2 = vec_2(:)';
        tmp = vec_1 - vec_2;
        dist(i,1) = sqrt(tmp * tmp');
        dist(i,2) = i-1;
    end
end