img = imread('/home/tatsch/PSPNet-Keras-tensorflow/example_images/cityscapes.png');
crop_size = 713;
resized = imresize(img, [crop_size crop_size], 'bilinear');
mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;

mean_rgb = [mean_r; mean_g; mean_b];
mean_rgb = reshape(mean_rgb, 1, 1, 3);

im_mean = repmat(mean_rgb, [crop_size, crop_size, 1]);

resized_f = single(resized);
demeaned = resized_f - im_mean;

mean_rc = mean(mean(demeaned(:, :, 1)));
mean_bc = mean(mean(demeaned(:, :, 2)));
mean_gc = mean(mean(demeaned(:, :, 3)));

figure;
imshow(demeaned)