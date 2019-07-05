function psnr=compute_psnr(im_gnd,im)
rmse=compute_rmse(im_gnd,im);
if max(im(:))>2
    psnr = 20*log10(255/rmse);
else
    psnr = 20*log10(1/rmse);
end
