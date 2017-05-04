clear
clc

img_list = readtable('list.txt');
folder = 'Database2k';
img_dir = 'images';

for i=1:height(img_list)
    img_file = table2array(img_list(i,1));
    img_file = img_file{1,1};
    class = table2array(img_list(i,2));
    
    command = sprintf('mkdir %s\\%d', folder, class);
    system(command);
    command = sprintf('cp %s\\%s %s\\%d\\%s', img_dir, img_file, folder, class, img_file);
    system(command);
end