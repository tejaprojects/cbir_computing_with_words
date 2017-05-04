clear
clc

folder = 'DataBase_2_5k';

file = fopen('classes_list_2_5k.txt', 'w');

for i=0:7
    curr_dir = sprintf('%s\\%d', folder, i);
    curr_dir_info = dir(sprintf('%s\\*.jpg', curr_dir));
    num_images = size(curr_dir_info, 1);
    fwrite(file, sprintf('%d\r\n', num_images));
end
fclose(file);
