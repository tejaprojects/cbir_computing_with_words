clear
clc

% file = fopen('dataset.txt', 'w');

img_path = 'DataBase_2k\';
img_list = table2array(readtable('images_list_2k.txt'));

dataset = zeros(length(img_list), 43);

imgx = img_list(:,1);
imgy = img_list(:,2);

parfor indx = 1:1:length(img_list)
    img_file = sprintf('%s\\%d\\%d.jpg', img_path, imgx(indx), imgy(indx));
    
	image = imread(img_file);
    img_info = imfinfo(img_file);
    if strcmp(img_info.ColorType, 'grayscale') == 1
        image = cat(3, image, image, image);
    end
    hsv = rgb2hsv(image);
    h = hsv(:,:,1); s = hsv(:,:,2); v = hsv(:,:,3);
    h = h(:); s = s(:); v = v(:);

    m_h = mean(h); m_s = mean(s); m_v = mean(v);
    std_h = std(h); std_s = std(s); std_v = std(v);
    skew_h = 0; skew_s = 0; skew_v = 0;
    for i=1:length(h)
        skew_h = skew_h + ((h(i)-m_h)^3);
        skew_s = skew_s + ((s(i)-m_s)^3);
        skew_v = skew_v + ((v(i)-m_v)^3);
    end
    skew_h = (skew_h/length(h))^(1/3);
    skew_s = (skew_s/length(h))^(1/3);
    skew_v = (skew_v/length(h))^(1/3);

    color_feature_vec_h = [m_h, std_h, skew_h];
    color_feature_vec_s = [m_s, std_s, skew_s];
    color_feature_vec_v = [m_v, std_v, skew_v];
    color_feature_vec = [color_feature_vec_h, color_feature_vec_s, color_feature_vec_v];

    [vertical, horizontal, diagonal] = RankletFilter(img_file, 5, 5);

    vertical = round(10*vertical)/10;
    horizontal = round(10*horizontal)/10;
    diagonal = round(10*diagonal)/10;

    rv = [-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,...
           0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];

    rh_v = zeros(1, 21);
    rh_h = zeros(1, 21);
    rh_d = zeros(1, 21);
    for i=1:21
        rh_v(i) = length(find(vertical == rv(i)));
        rh_h(i) = length(find(horizontal == rv(i)));
        rh_d(i) = length(find(diagonal == rv(i)));
    end

    mu_v = mean(rh_v);
    mu_h = mean(rh_h);
    mu_d = mean(rh_d);

    sigma_v = std(rh_v);
    sigma_h = std(rh_h);
    sigma_d = std(rh_d);

    mc_v = 0; mc_h = 0; mc_d = 0;
    cv_v = 0; cv_h = 0; cv_d = 0;
    for i=1:21
        mc_v = mc_v + abs(rv(i)*rh_v(i)-mu_v);
        mc_h = mc_h + abs(rv(i)*rh_h(i)-mu_h);
        mc_d = mc_d + abs(rv(i)*rh_d(i)-mu_d);

        cv_v = cv_v + (rv(i)-mu_v)*(rv(i)-mu_v)*rh_v(i);
        cv_h = cv_h + (rv(i)-mu_h)*(rv(i)-mu_h)*rh_h(i);
        cv_d = cv_d + (rv(i)-mu_d)*(rv(i)-mu_d)*rh_d(i);
    end

    vertical_pad = padarray(vertical, [1,1], 'replicate');
    horizontal_pad = padarray(horizontal, [1,1], 'replicate');
    diagonal_pad = padarray(diagonal, [1,1], 'replicate');

    ce_v = 0; ce_h = 0; ce_d = 0;
    un_v = 0; un_h = 0; un_d = 0;
    fdm_v = 0; fdm_h = 0; fdm_d = 0;
    sdm_v = 0; sdm_h = 0; sdm_d = 0;
    fidm_v = 0; fidm_h = 0; fidm_d = 0;
    sidm_v = 0; sidm_h = 0; sidm_d = 0;
    for i=1:21
        for j=1:21
            tmp = rcm(vertical_pad,i,j,rv);
            ce_v = ce_v + tmp*log(tmp);
            un_v = un_v + tmp*tmp;
            fdm_v = fdm_v + abs(i-j)*tmp;
            sdm_v = sdm_v + (i-j)*(i-j)*tmp;
            fidm_v = fidm_v + tmp/(1+abs(i-j));
            sidm_v = sidm_v + tmp/(1+(i-j)*(i-j));

            tmp = rcm(horizontal_pad,i,j,rv);
            ce_h = ce_h + tmp*log(tmp);
            un_h = un_h + tmp*tmp;
            fdm_h = fdm_h + abs(i-j)*tmp;
            sdm_h = sdm_h + (i-j)*(i-j)*tmp;
            fidm_h = fidm_h + tmp/(1+abs(i-j));
            sidm_h = sidm_h + tmp/(1+(i-j)*(i-j));

            tmp = rcm(diagonal_pad,i,j,rv);
            ce_d = ce_d + tmp*log(tmp);
            un_d = un_d + tmp*tmp;
            fdm_d = fdm_d + abs(i-j)*tmp;
            sdm_d = sdm_d + (i-j)*(i-j)*tmp;
            fidm_d = fidm_d + tmp/(1+abs(i-j));
            sidm_d = sidm_d + tmp/(1+(i-j)*(i-j));
        end
    end

    edrcm1_v = 0; edrcm1_h = 0; edrcm1_d = 0;
    for i=9:13
        for j=9:13
            tmp = rcm(vertical_pad,i,j,rv);
            edrcm1_v = edrcm1_v + tmp;

            tmp = rcm(horizontal_pad,i,j,rv);
            edrcm1_h = edrcm1_h + tmp;

            tmp = rcm(diagonal_pad,i,j,rv);
            edrcm1_d = edrcm1_d + tmp;
        end
    end

    edrcm2_v = 0; edrcm2_h = 0; edrcm2_d = 0;
    for i=7:15
        for j=7:15
            tmp = rcm(vertical_pad,i,j,rv);
            edrcm2_v = edrcm2_v + tmp;

            tmp = rcm(horizontal_pad,i,j,rv);
            edrcm2_h = edrcm2_h + tmp;

            tmp = rcm(diagonal_pad,i,j,rv);
            edrcm2_d = edrcm2_d + tmp;
        end
    end
    edrcm2_v = edrcm2_v - edrcm1_v;
    edrcm2_h = edrcm2_h - edrcm1_h;
    edrcm2_d = edrcm2_d - edrcm1_d;

    edrcm3_v = 0; edrcm3_h = 0; edrcm3_d = 0;
    for i=3:19
        for j=3:19
            tmp = rcm(vertical_pad,i,j,rv);
            edrcm3_v = edrcm3_v + tmp;

            tmp = rcm(horizontal_pad,i,j,rv);
            edrcm3_h = edrcm3_h + tmp;

            tmp = rcm(diagonal_pad,i,j,rv);
            edrcm3_d = edrcm3_d + tmp;
        end
    end
    edrcm3_v = edrcm3_v - edrcm1_v - edrcm2_v;
    edrcm3_h = edrcm3_h - edrcm1_h - edrcm2_h;
    edrcm3_d = edrcm3_d - edrcm1_d - edrcm2_d;

    texture_feature_vec_v = [mc_v, cv_v, ce_v, un_v, fdm_v, sdm_v, fidm_v, sidm_v, edrcm1_v, edrcm2_v, edrcm3_v];
    texture_feature_vec_h = [mc_h, cv_h, ce_h, un_h, fdm_h, sdm_h, fidm_h, sidm_h, edrcm1_h, edrcm2_h, edrcm3_h];
    texture_feature_vec_d = [mc_d, cv_d, ce_d, un_d, fdm_d, sdm_d, fidm_d, sidm_d, edrcm1_d, edrcm2_d, edrcm3_d];
    texture_feature_vec = [texture_feature_vec_v, texture_feature_vec_h, texture_feature_vec_d];

    feature_vec = [color_feature_vec, texture_feature_vec];
    status = isnan(feature_vec);
    feature_vec(status) = zeros(1, sum(status));

%     fwrite(file, sprintf('%f', feature_vec(1)));
%     for i=2:length(feature_vec)
%         fwrite(file, sprintf(',%f', feature_vec(i)));
%     end
%     fwrite(file, sprintf(',%d\r\n', indx-1));
    
    dataset(indx, :) = [feature_vec, indx-1];
    disp(sprintf('Finished: %d', indx));
end

for i=1:length(dataset)
    for j=1:43
        dataset(i,j) = abs(dataset(i,j));
    end
end