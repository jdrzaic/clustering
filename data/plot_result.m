function [] = plot_result(data, x)
first_x = [];
first_y = [];
second_x = [];
second_y = [];
third_x = [];
third_y = [];

for i = 1:130
if x(i,1) == 1
first_x = [first_x data(1, i)];
first_y = [first_y data(2, i)];
end
if x(i,2) == 1
second_x = [second_x data(1, i)];
second_y = [second_y data(2, i)];
end
if x(i,3) == 1
third_x = [third_x data(1,i)];
third_y = [third_y data(1,i)];
end
end
plot(first_x, first_y, '*')
%plot(second_x, second_y, '*')
%plot(third_x, third_y, '*')
end
