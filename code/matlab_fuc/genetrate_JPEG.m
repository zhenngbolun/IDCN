dataset = 'WIN143';
scale = 3;
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
if dataset == "DIV2K"
    org_image_dir = 'F:\\dataset\\DIV2K\\\DIV2K_train_HR\\';
    test_image_dir ='F:\dataset\DIV2K\DIV2K_JPEG\\y\\';
    test_image_dir_10 = 'F:\\dataset\\DIV2K\\\QF10\\y\\';
    test_image_dir_20 = 'F:\\dataset\\DIV2K\\\QF20\\y\\';
    test_image_dir_40 = 'F:\\dataset\\DIV2K\\\QF40\\y\\';
    file_list = dir([org_image_dir,'*.png']);
end
if dataset ~= "DIV2K"
    %test_image_dir = ['F:\\code\\code\\test_image\\',dataset,'\\'];
    test_image_dir = [org_image_dir,'y\\'];
end
len = length(file_list);
for i = 1:len
    name = file_list(i).name;
    l = length(name);
    name0 = name(1:l-4);
    im1 = imread([org_image_dir,name]);
    if size(im1, 3) == 3
        im = rgb2ycbcr(im1);
        im1 = im(:,:,1);
    end
    imwrite(im1,[test_image_dir,name0,'.png']);
    %imwrite(im1,[test_image_dir,name0,'.jpg'],'jpg','Quality',20);
    %imwrite(im1,[test_image_dir_40,name0,'.jpg'],'jpg','Quality',40);
end