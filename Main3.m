function [x, y, H3] = Main3()

addpath("/home/icaros/grasp")
load R_table_gt -mat
H3 = rewardMat(R_table_full);
[x, y] = meshgrid(1:285, 1:285);

end