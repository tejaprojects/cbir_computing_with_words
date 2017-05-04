function [class_num] = which_class(indx, classes)
    for i = 1:length(classes)
        if indx <= classes(i)
            class_num = i-1;
            break;
        end
    end
end