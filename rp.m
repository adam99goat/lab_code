%D_test = disp_read('data\img_2.png');
tau1 = [1 0];
tau2 = [2 0];
tau3 = [3 0];

e_num = {'1','2','3','6','7'};
k_num = {'0','1','2','3','4'};
k_num1 = {'1','2','3','4','5'};

top_addr1='C:\Users\HUST\Desktop\result process\'; %%放result process文件夹的位置
file_addr={'result_35','result25','result25_output1','moredata'}; %%result process文件夹里关于result的子文件夹
read_addr=cell(1,4);
gt_addr='C:\Users\HUST\Desktop\result process\GT\';  %%放GT文件夹的位置
save_addr=cell(1,4);  %%存图的位置，事先在桌面建好文件夹
top_addr2='C:\Users\HUST\Desktop\';
for i=1:4
    read_addr{i}=[top_addr1,file_addr{i}];%%读est图的地址
    save_addr{i}=[top_addr2,file_addr{i}];%%存误差图的地址
end

bf=[1035.033*4.14353*ones(1,3),1086.769*4.36414*ones(1,2)];%%分别对应1、2、3、6、7
title_list={'result\_35','result25','result25\_output1','moredata'};%%误差图中的标题

for j=1:4%%三个result
    for k=1:5%%五个estimate
        for i=1:5%%五个key
            D_est1 = imread([read_addr{j},'\e',e_num{k},'\left_depth_map',k_num{i},'.tiff']);
            D_gt = imread([gt_addr,'d',e_num{k},'\k',k_num1{i},'\left_depth_map.tiff']);
            Left = imread([gt_addr,'d',e_num{k},'\k',k_num1{i},'\Left_Image.png']);
            
            
            D_gt = bf(k)./D_gt;
            D_est1 = bf(k)./D_est1;
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            d_err1 = disp_error(D_gt,D_est1,tau2,192);
            D_err1 = disp_error_image(D_gt,D_est1,tau2);
            figure(10)
            subplot(3,2,4),imshow(D_err1);
            
            title(sprintf('2-pix Disparity Error: %.2f %%',d_err1*100));
            set(gca,'OuterPosition', [0.48,0.31,0.45,0.4]);
            
            D_test_color0=disp_to_color(D_gt,192);
            subplot(3,2,1),imshow(D_test_color0)
            %title('Pic: d1k5','position',[650,1125]);%%%%%%%%%%%%%%%%%
            set(gca,'OuterPosition', [0,0.65,0.45,0.4]);
            
            
            D_test_color1=disp_to_color(D_est1,192);
            subplot(3,2,3),imshow(D_test_color1)
            title(title_list{j})
            set(gca,'OuterPosition', [0,0.31,0.45,0.4]);
            
            d_err0 = disp_error(D_gt,D_est1,tau1,192);
            D_err0 = disp_error_image(D_gt,D_est1,tau1);
            subplot(3,2,2),imshow(D_err0)
            title(sprintf('1-pix Disparity Error: %.2f %%',d_err0*100));
            set(gca,'OuterPosition', [0.48,0.65,0.45,0.4]);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            d_err2 = disp_error(D_gt,D_est1,tau3,192);
            D_err2 = disp_error_image(D_gt,D_est1,tau3);
            %figure(1),imshow([disp_to_color([D_est;D_gt]);D_err]);
            subplot(3,2,6),imshow(D_err2);
            title(sprintf('3-pix Disparity Error: %.2f %%',d_err2*100));
            set(gca,'OuterPosition', [0.48,-0.03,0.45,0.4]);
            
            D_test_color2=disp_to_color(D_est1,192);
            subplot(3,2,5),imshow(D_test_color2)
            title(title_list{j})
            set(gca,'OuterPosition', [0,-0.03,0.45,0.4]);
            
            saveas(gcf, [save_addr{j},'\d',e_num{k},'k',k_num{i},'.png'])
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%
% 
% imwrite( D_test_color0, ['E:\Ph.D\nianhui pic\compare\alone\freeze\depth_d',e_num,'k',k_num,'.png'])
% imwrite( D_err0, ['E:\Ph.D\nianhui pic\compare\alone\freeze\error_d',e_num,'k',k_num,'.png'])
% 
% imwrite( D_test_color1,['E:\Ph.D\nianhui pic\compare\alone\nomask\depth_d',e_num,'k',k_num,'.png'])
% imwrite( D_err1, ['E:\Ph.D\nianhui pic\compare\alone\nomask\error_d',e_num,'k',k_num,'.png'])
% 
% imwrite( D_test_color2, ['E:\Ph.D\nianhui pic\compare\alone\more\depth_d',e_num,'k',k_num,'.png'])
% imwrite( D_err2,['E:\Ph.D\nianhui pic\compare\alone\more\error_d',e_num,'k',k_num,'.png'])
