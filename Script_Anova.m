% Script_6 Askhsh 6

close all;clear all;clc

addpath('C:\Users\Michael\Desktop\Matlab\PRTools')

% we create two one-dimensional datasets with mean zero and std one and 500
%   samples
set_1=mvnrnd(0,1,500);
set_2=mvnrnd(0,1,500);

% and we call the anova1() function
[p anovatab stats]=anova1([set_1 set_2]);