clear
clc

folder = 'Database_2_5k';

file = fopen('images_list_2_5k.txt', 'w');
file_jpg = fopen('jpg_images_list_2_5k.txt', 'w');

for i=0:7
    curr_dir = sprintf('%s\\%d', folder, i);
    curr_dir_info = dir(sprintf('%s\\*.jpg', curr_dir));
    num_images = size(curr_dir_info, 1);
    for j = 1:num_images
        img_file = sprintf('%s\\%s', curr_dir, curr_dir_info(j).name);
        command = sprintf('cp %s %s\\%d.jpg', img_file, curr_dir, j-1);
        system(command);
        command = sprintf('rm %s', img_file);
        system(command);
        fwrite(file, sprintf('%s\\%d\r\n', curr_dir, j-1));
        fwrite(file_jpg, sprintf('%s\\%d.jpg\r\n', curr_dir, j-1));
    end
end
fclose(file);
fclose(file_jpg);
