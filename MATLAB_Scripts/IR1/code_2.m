folder = 'DataBase2';

file = fopen('classes_list.txt', 'w');

counts = zeros(1, 15);
for i=0:9
    curr_dir = sprintf('%s\\%d', folder, i);
    curr_dir_info = dir(sprintf('%s\\*.jpg', curr_dir));
    num_images = size(curr_dir_info, 1);
    counts(i+1) = num_images;
    fwrite(file, sprintf('%d\r\n', counts(i+1)));
end
fclose(file);
