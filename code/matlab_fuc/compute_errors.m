function errors=compute_errors(im_gnd , im)

if size(im_gnd, 3) == 3,
    im_gnd = rgb2ycbcr(im_gnd);
    im_gnd = im_gnd(:, :, 1);
end

if size(im, 3) == 3,
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end

im_gnd = double(im_gnd);
im = double(im);

if max(im(:)) < 2
    im = im * 255;
end
if max(im_gnd(:)) < 2
    im_gnd = im_gnd * 255;
end

PSNR = compute_psnr(im_gnd, im);
PSNR_B = compute_psnrb(im_gnd, im);
BEF = compute_bef(im);
SSIM = ssim_index(im_gnd, im, [0.01 0.03], ones(8));

errors = [PSNR ; PSNR_B; BEF; SSIM];

fprintf(' PSNR: %f;\n PSNR_B: %f;\n BEF: %f;\n SSIM: %f;\n\n', PSNR, PSNR_B, BEF, SSIM);

