dataset = 'B200';
scale = 8;
mode = 'JPEG';
if dataset == "WIN143"
    org_image_dir = 'F:\\dataset\\WIN143_resize\\';
    file_list = dir([org_image_dir,'*.png']);
end
if dataset == "LIVE1"
    org_image_dir = 'F:\\dataset\\LIVE1\\';
    file_list = dir([org_image_dir,'*.bmp']);
end
if dataset == "B200"
    org_image_dir = 'F:\\dataset\\BSDS500\\images\\test\\';
    file_list = dir([org_image_dir,'*.jpg']);
end
if dataset == "B100"
    org_image_dir = 'F:\\dataset\\BSDS500\\images\\val\\';
    file_list = dir([org_image_dir,'*.jpg']);
end
if dataset == "Set5"
    org_image_dir = 'F:\\dataset\\Set5\\';
    file_list = dir([org_image_dir,'*.bmp']);
end
if dataset == "Set14"
    org_image_dir = 'F:\\dataset\\Set14\\';
    file_list = dir([org_image_dir,'*.bmp']);
end
if dataset == "Classic5"
    org_image_dir = 'F:\\dataset\\classic5\\';
    file_list = dir([org_image_dir,'*.bmp']);
end
if dataset == "B68"
    org_image_dir = 'F:\\dataset\\BSD68\\';
    file_list = dir([org_image_dir,'*.png']);
end
test_image_dir = ['F:\\code\\code\\test_image\\',dataset,'\\'];
%test_image_dir =[org_image_dir,'X',num2str(scale),'\\'];
len = length(file_list);
crop = true;

psnrs = 0;
psnrbs = 0;
ssims = 0;
count = 0;
for i = 1:len
    name = file_list(i).name;
    l = length(name);
    name0 = name(1:l-4);
    im1 = imread([org_image_dir,name]);
    im2 = imread([test_image_dir,name0,'.bmp']); 
    if crop
        [h,w,c] = size(im1);
        h = h - mod(h,scale);
        w = w - mod(w,scale);
        im1 = im1(1:h,1:w,:);
        im2 = im2(1:h,1:w,:);
    end
    
    if mode == "JPEG"
        if size(im1,3)==3
            im = rgb2ycbcr(im1);
            im1 = im(:,:,1);
        end

        if size(im2,3)==3
            im = rgb2ycbcr(im2);
            im2 = im(:,:,1);
        end
        psnr_ = compute_psnr(im1, im2);
        
        ssim_ = 0;
        %ssim_ = ssim(im1, im2);
        psnrb_ = 0;
        for c = 1:1        
            ssim_ = ssim_ + ssim_index(im1(:,:,c), im2(:,:,c), [0.01,0.03], ones(8));            
            psnrb_ = psnrb_ + compute_psnrb(im1(:,:,c), im2(:,:,c));
        end
        ssim_ = ssim_ / 1;
        psnrb_ = psnrb_ / 1;
        fprintf('%s: psnr - %.2f, ssim - %.4f, psnrb - %.2f\r',name0, psnr_, ssim_, psnrb_);
        psnrbs = psnrbs + psnrb_;
    else
        if size(im1,3)==3
            im = rgb2ycbcr(im1);
            im1 = im(:,:,1);
        end

        if size(im2,3)==3
            im = rgb2ycbcr(im2);
            im2 = im(:,:,1);
        end
        im1 = im1(scale+1:h-scale,scale+1:w-scale);
        im2 = im2(scale+1:h-scale,scale+1:w-scale);
        psnr_ = compute_psnr(im1, im2);
        ssim_ = ssim(im1, im2);
        fprintf('%s: psnr - %.2f, ssim - %.4f\r',name0, psnr_, ssim_);
    end
    
    psnrs = psnrs + psnr_;
    ssims = ssims + ssim_;
    
    
end

psnrs = psnrs / len
ssims = ssims / len
psnrbs = psnrbs/ len