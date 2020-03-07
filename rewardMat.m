function [H] = rewardMat(R_table_full)

H = zeros(285);
for i = 1:285
    for j = 1:285
        temp = cell2mat(R_table_full(i,j));
        H(i, j) = temp(1);
    end
end
    

end

