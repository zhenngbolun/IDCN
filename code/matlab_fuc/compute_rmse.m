function [rmse] = compute_rmse(im1, im2)

imdff = double(im1) - double(im2);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));