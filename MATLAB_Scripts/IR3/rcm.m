function [count] = rcm(in,I,J,rv)
    [m,n] = size(in);
    count_0 = 0; count_45 = 0;count_90 = 0; count_135 = 0;
    for i=2:m-1
        for j=2:n-1
            if in(i,j) == rv(I)
                if in(i,j+1) == rv(J)
                    count_0 = count_0 + 1;
                end
                if in(i-1,j+1) == rv(J)
                    count_45 = count_45 + 1;
                end
                if in(i-1,j) == rv(J)
                    count_90 = count_90 + 1;
                end
                if in(i-1,j-1) == rv(J)
                    count_135 = count_135 + 1;
                end 
            end
        end
    end
    count = (count_0 + count_45 + count_90 + count_135)/4;
end