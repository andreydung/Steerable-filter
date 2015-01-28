clear all;
close all;
clc;

im = double(imread('01.tif'));

bands = get_pyramid_noSub(im, 2, 4);



