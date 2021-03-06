
close all;
clear;
clc;

% g1 = gpuDevice(2);
% reset(g1);
g2 = gpuDevice(3);
reset(g2);

%% DNN_Training

load('Dataset.mat');    % 效果�?�?
% % 
% % for i = 1:(size(Pos, 2)-1)
% %     error_firstorder(:, i) = Pos(2:7, i+1) - Pos(2:7, i);
% % end
% % error_firstorder = [error_firstorder error_firstorder(:, end)];
% % 
% % for i = 1:(size(Pos, 2)-2)
% %     error_secondorder(:, i) = Pos(2:7, i+2) - Pos(2:7, i);
% % end
% % error_secondorder = [error_secondorder error_secondorder(:, (end-1):end)];
% % 
% % % for i = 1:(size(Pos, 2)-3)
% % %     error_thirdorder(:, i) = Pos(2:7, i+3) - Pos(2:7, i);
% % % end
% % % error_thirdorder = [error_thirdorder error_thirdorder(:, (end-2):end)];
% % 
% % error_data = [error_firstorder; error_secondorder];

% % % data_X = [Pos(2:6, :); Pos(2:6, 2:end) Pos(2:6, end); Pos(2:6, :) Pos(2:6, (end-1):end)];
% % % data_X = [Pos(6, :); Pos(6, 2:end) Pos(6, end); Pos(6, 3:end) Pos(6, (end-1):end)];
% % % data_X = [Pos(2:7, :); error_data];
% % 
data_num = 5120;
data_X = Pos(2:7, :);

% data_X = [Pos(2:7, :); Pos(2:7, 2:end) Pos(2:7, end); Pos(2:7, 3:end) Pos(2:7, (end-1):end)];
data_Y = Pos(8:12, :); 

data_x = data_X(:, 1:1:end);
data_y = data_Y(:, 1:1:end);
% data_x = data_X(:, :);
% data_y = data_Y(:, :);
%-------------------------------------------------------------------------------------

%%-------------------------------------------------------------------------------------
% ��һ�� [y,ps] = mapminmax(x,ymin,ymax)��
[norm_x, input] = mapminmax(data_x, -1, 1);
[norm_y, output] = mapminmax(data_y, -1, 1);
%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% ˳�����? 840��Ϊ��ѵ���������? 240���ڱ仯���? ����120���ڲ������?
% ������÷���? [trainV,valV,testV] = dividevec(p,t,valPercent,testPercent)
testPercent = 0.05;
validationPercent = 0.15;
[trainVar, validationVar, testVar] = dividevec(norm_x(:, 1:(end-data_num)), norm_y(:, 1:(end-data_num)), validationPercent, testPercent);

Predition_x = norm_x(:, (end-data_num+1):end);
Predition_y = norm_y(:, (end-data_num+1):end);
%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% �������ṹ�������������?0
rand('state',80);

layer1 = 25;
layer2 = 30;
layer3 = 30; 
layer4 = 35;
% layer5 = 30;
% layer6 = 32;
% layer7 = 32;
% layer8 = 32;

layer_nodes = [layer1, layer2, layer3, layer4];
% layer_nodes = [layer1, layer2, layer3];
% layer_nodes = [layer1, layer2];
% layer_nodes = [layer1, layer2, layer3, layer4, layer5];


net = feedforwardnet(layer_nodes, 'trainscg' );

% net.LW{2, 2}= net.LW{2, 2}*0.5;                       % ��ʼ��Ȩ������
% net.b{2, 1} =net.b{2, 1}*0.5;                             % ��ʼ��ƫ��
net.trainParam.epochs = 2000;                             % �������ѵ������?2000��
net.trainParam.goal = 1e-11;
net.trainParam.sigma = 100.0e-5;
net.trainParam.lambda = 100.0e-7;

% net.trainParam.lr = 0.1;
% net.trainParam.mc= 0.95;                            % Momentum constant. 

net.performFcn = 'mse';


% [net, tr] = train(net, trainVar.P, trainVar.T, [], [], validationVar, testVar.T);
[net, tr] = train(net, trainVar.P, trainVar.T, 'UseGPU', 'only', 'ShowResources','yes');
% [net, tr] = train(net, trainVar.P, trainVar.T , 'UseParallel','yes', 'UseGPU', 'only', 'ShowResources','yes');

% [net, tr] = train(net, trainVar.P, trainVar.T);

%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% ѵ����ɾͿ��Ե���sim()������з�����?
% [norm_trainy, trainPerf] = sim(net, trainVar.P, [], [], trainVar.T);
[norm_trainy, trainPerf] = sim(net, trainVar.P, 'UseGPU','only');
[norm_validationy, validsationPerf] = sim(net, validationVar.P, 'UseGPU','only');
[norm_testy, testPerf] = sim(net, testVar.P , 'UseGPU','only');

[norm_predictiony, predictionPerf] = sim(net, Predition_x , 'UseGPU','only');

perf = perform(net, testVar.T, norm_testy);

% [norm_trainy, trainPerf] = sim(net, trainVar.P );
% [norm_validationy, validsationPerf] = sim(net, validationVar.P);
% [norm_testy, testPerf] = sim(net, testVar.P);
% 
% [norm_predictiony, predictionPerf] = sim(net, Predition_x );
%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% ����һ������
output_trainnet = mapminmax('reverse', norm_trainy, output);
output_train = mapminmax('reverse', trainVar.T, output);
output_validationnet = mapminmax('reverse', norm_validationy, output);
output_validation = mapminmax('reverse', validationVar.T, output);
output_testnet = mapminmax('reverse', norm_testy, output);
output_test = mapminmax('reverse', testVar.T, output);

output_preditionnet = mapminmax('reverse', norm_predictiony, output);
output_predition = mapminmax('reverse', Predition_y, output);



%%

layer21 = 15;
layer22 = 15;
layer23 = 15; 
layer24 = 15;
% layer25 = 30;

layer_nodes2 = [layer21, layer22, layer23, layer24];
% layer_nodes2 = [layer21, layer22, layer23];
% layer_nodes2 = [layer21, layer22];
% layer_nodes2 = [layer21, layer22, layer23, layer24, layer25];


net2 = feedforwardnet(layer_nodes2, 'trainscg' );

% net.LW{2, 2}= net.LW{2, 2}*0.5;                       % ��ʼ��Ȩ������
% net.b{2, 1} =net.b{2, 1}*0.5;                             % ��ʼ��ƫ��
net2.trainParam.epochs = 2000;                             % �������ѵ������?2000��
net2.trainParam.goal = 1e-11;
net2.trainParam.sigma = 100.0e-5;
net2.trainParam.lambda = 100.0e-7;


% [net, tr] = train(net, trainVar.P, trainVar.T, [], [], validationVar, testVar.T);
[net2, tr2] = train(net2, trainVar.P(4:end, : ), trainVar.T(4:end, : ), 'UseGPU', 'only', 'ShowResources','yes');
% [net, tr] = train(net, trainVar.P, trainVar.T , 'UseParallel','yes', 'UseGPU', 'only', 'ShowResources','yes');

% [net, tr] = train(net, trainVar.P, trainVar.T);

%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% ѵ����ɾͿ��Ե���sim()������з�����?
% % [norm_trainy, trainPerf] = sim(net, trainVar.P, [], [], trainVar.T);
% [norm_trainy, trainPerf] = sim(net, trainVar.P, 'UseGPU','only');
% [norm_validationy, validsationPerf] = sim(net, validationVar.P, 'UseGPU','only');
% [norm_testy, testPerf] = sim(net, testVar.P , 'UseGPU','only');

[norm_predictiony2, predictionPerf2] = sim(net2, Predition_x(4:end, : ) , 'UseGPU','only');

% perf = perform(net2, testVar.T, norm_testy);

% [norm_trainy, trainPerf] = sim(net, trainVar.P );
% [norm_validationy, validsationPerf] = sim(net, validationVar.P);
% [norm_testy, testPerf] = sim(net, testVar.P);
% 
% [norm_predictiony, predictionPerf] = sim(net, Predition_x );
%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% ����һ������
% output_trainnet = mapminmax('reverse', norm_trainy, output);
% output_train = mapminmax('reverse', trainVar.T, output);
% output_validationnet = mapminmax('reverse', norm_validationy, output);
% output_validation = mapminmax('reverse', validationVar.T, output);
% output_testnet = mapminmax('reverse', norm_testy, output);
% output_test = mapminmax('reverse', testVar.T, output);
alpha = 0;
norm_matrix = [zeros(3, data_num); norm_predictiony2];
output_preditionnet2 = mapminmax('reverse', norm_matrix, output);
% output_predition = mapminmax('reverse', Predition_y, output);
output_preditionnet(4:end, : ) = alpha * output_preditionnet(4:end, : ) + (1 - alpha) * output_preditionnet2(4:end, : );
%-------------------------------------------------------------------------------------

%%------------------------------------------------------------------------------------
% % ������������
% error_train = output_train - output_trainnet;
% error_test = output_test - output_testnet;
% mse_train = mse(net, output_train, output_trainnet);
% mse_test = mse(net, output_test, output_testnet);

% mse_train = mse(error_train);
% mse_test = mse(error_test);
%-------------------------------------------------------------------------------------

%-------------------------------------------------------------------------------------
% ��ݷ������ͼ
% plot(data(2,:), data(3,:), '+')
% figure('position',[10,10,1000,800],'name','DNN for robots');
% for n = 1:5
%     subplot(2, 3, n);
%     hold on;
%     plot(1:size(output_test,2), output_test(n, :),'b');
%     hold on;
%     plot(1:size(output_test,2), output_testnet(n, :),'rx');
%     xlabel('t');
%     ylabel(['Joint_' num2str(n)]);
%     title('b��ʾ��ʵֵ��r��ʾԤ��ֵ');
% end

% save Prediction_datanet output_preditionnet
% save Prediction_data output_predition
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
% error_test = output_test - output_testnet;
% figure('position',[10,10,1000,800],'name','Error for robots');
% for n = 1:6
%     subplot(2, 3, n);
%     hold on;
%     plot(data(1, 1:120), error_test(n, :),'b');
%     xlabel('t');
%     ylabel(['Joint_' num2str(n)]);
%     title('net error');
% end

% sprintf('Mse_train = %f,   Mse_test = %f', mse_train, mse_test)


%% Data_Running

% Set general condition 
Ts=0.001;                                                           % Sample Period: 1ms 
VlimitX = 3; VlimitY = 3; VlimitZ = 3; VlimitA = 2; VlimitC = 2;    % Drives' input saturation value

InterpolationData=load('InterpInfo_Zhao_Fan.mat');
% InterpolationData=load('InterpInfo_Zhao_Blade.mat');
ToolTipPos=InterpolationData.interpcor(:,1:3);

X=ToolTipPos(:,1);                                                      % Set the parameters for C-MEX S function for hunting the foot point
Y=ToolTipPos(:,2);
Z=ToolTipPos(:,3);
U=InterpolationData.interpcor(:,9);
V=InterpolationData.interpcor(:,7);
A=InterpolationData.interpcor(:,8);

%ToolOrienPos=InterpolationData.interpcor(:,4:6);
% Indentified the orientation vectors
lenOrienVector=length(U);
ToolOrienPos=zeros(lenOrienVector,3);
for ii=1:lenOrienVector
    TempOrienVector=InterpolationData.interpcor(ii,4:6);
    ToolOrienPos(ii,1:3)=InterpolationData.interpcor(ii,4:6)/norm(TempOrienVector);
end

I=ToolOrienPos(:,1);
J=ToolOrienPos(:,2);
K=ToolOrienPos(:,3);

DriveCommands =InverseKinematics_DH(ToolTipPos,ToolOrienPos);
HomePosDrive=DriveCommands(1,:);                                        % Set the home positon for motion control

% % load('Prediction_data.mat');
% % Xcommands = output_predition(1,:);           
% % Ycommands = output_predition(2,:);          
% % Zcommands = output_predition(3,:);  
% % Acommands = output_predition(4,:);  
% % Ccommands = output_predition(5,:);

% load('Prediction_datanet.mat');
Pre_commands(1, :) = output_preditionnet(1,:) + HomePosDrive(1);
Pre_commands(2, :) = output_preditionnet(2,:) + HomePosDrive(2);
Pre_commands(3, :) = output_preditionnet(3,:) + HomePosDrive(3);
Pre_commands(4, :) = output_preditionnet(4,:) + HomePosDrive(4);
Pre_commands(5, :) = output_preditionnet(5,:) + HomePosDrive(5);

HomePosCommonds=Pre_commands(:, 1);                                        % Set the home positon for motion control
Xcommands = Pre_commands(1,:) - HomePosCommonds(1);            
Ycommands = Pre_commands(2,:) - HomePosCommonds(2);                 
Zcommands = Pre_commands(3,:) - HomePosCommonds(3);         
Acommands = Pre_commands(4,:) - HomePosCommonds(4);         
Ccommands = Pre_commands(5,:) - HomePosCommonds(5);       

% Set the plant X,Y,Z,A,Z parameters
[PlantXPara,PlantYPara,PlantZPara,PlantAPara,PlantCPara]=PlantParameters(Ts);
PlantXNum=[PlantXPara.b1 PlantXPara.b0];
PlantXDen=[1 PlantXPara.a1 PlantXPara.a0];
PlantYNum=[PlantYPara.b1 PlantYPara.b0];
PlantYDen=[1 PlantYPara.a1 PlantYPara.a0];
PlantZNum=[PlantZPara.b1 PlantZPara.b0];
PlantZDen=[1 PlantZPara.a1 PlantZPara.a0];
PlantANum=[PlantAPara.b1 PlantAPara.b0];
PlantADen=[1 PlantAPara.a1 PlantAPara.a0];
PlantCNum=[PlantCPara.b1 PlantCPara.b0];
PlantCDen=[1 PlantCPara.a1 PlantCPara.a0];


% Simulate the model
sim('Controller_PID');

%% plot the trajectory
% Plot tool tip contour error and tool orientation contour error
% load('ContourTipOri_PID.mat');
load('ContourTipOri01.mat');
constant = 5000;
ts =  0 : Ts : 0.001*constant-0.001;
figure(1);
Tip_PID = Proposed(2, 1:constant) * 1e3;
plot(ts, Tip_PID, 'b-', 'linewidth', 2);
hold on
load('ContourTipOri.mat');
Tip_DNN = Proposed(2,:) * 1e3;
plot(ts, Tip_DNN(:, 1:constant), 'r:', 'linewidth', 2);
legend('PID', 'DNNPID');
xlim([0 5]);
xlabel('Time(ms)');
ylabel('Contour error [um]');
title('Tool tip contour error');

figure(2);
% load('ContourTipOri_PID.mat');
load('ContourTipOri01.mat');
Tool_PID = Proposed(3, 1:constant) * 1e3;
plot(ts, Tool_PID, 'b-', 'linewidth', 2);
hold on
load('ContourTipOri.mat');
Tool_DNN = Proposed(3,:) * 1e3;
plot(ts, Tool_DNN(:, 1:constant), 'r:', 'linewidth', 2);
legend('PID', 'DNNPID');
xlim([0 5]);
xlabel('Time(ms)');
ylabel('Contour error [mrad]');
title('Tool orientation contour error');

%% Plot the desired trajectory and the predicted trajectory
figure1 = figure(3);
axes1 = axes('Parent',figure1);
view(axes1,[47.5 60]);
hold(axes1,'on');

%  ToolTipOri_Desired
load('ToolTipOri_Desired.mat');
ToolTipOri_Desired = ActualPos( 2:7 ,  : );
plot3(ToolTipOri_Desired( 1 , : ),ToolTipOri_Desired( 2 , : ),ToolTipOri_Desired( 3 , : ),'b','Linewidth',2);
hold on
% ToolTipOri_PID
load('ToolTipOri_PID.mat');
ToolTipOri_PID = ActualPos( 2:7 ,  :  );
plot3(ToolTipOri_PID( 1 , : ),ToolTipOri_PID( 2 , : ),ToolTipOri_PID( 3 , : ),'c--','Linewidth',2);
hold on
% ToolTipOri_DNN
output_commands = [];
for ss=1:(size(output_preditionnet, 2))
    output_commands(1,ss)=output_preditionnet(1,ss) + HomePosDrive(1);
    output_commands(2,ss)=output_preditionnet(2,ss) + HomePosDrive(2);
    output_commands(3,ss)=output_preditionnet(3,ss) + HomePosDrive(3);
    output_commands(4,ss)=output_preditionnet(4,ss) + HomePosDrive(4);
    output_commands(5,ss)=output_preditionnet(5,ss) + HomePosDrive(5);
end
[Pa,Oa] = ForwardKinematics_DH(output_commands');
plot3(Pa( : , 1 ),Pa( : , 2 ),Pa( : , 3 ),'g','Linewidth',2);
hold on
% ToolTipOri_DNNPID
load('ToolTipOri_DNNPID.mat');
ToolTipOri_DNNPID = Proposed( 2:7 , : );
plot3(ToolTipOri_DNNPID( 1 , : ),ToolTipOri_DNNPID( 2 , : ),ToolTipOri_DNNPID( 3 , : ),'r--','Linewidth',2);

legend('Desired', 'PID', 'DNN', 'DNNPID')
xlabel('X [mm]')
ylabel('Y [mm]')
zlabel('Z [mm]')
title('Tool tip position spline');
set(gca,'FontSize',14);
hold off;
% 
% % Orientation
figure2 = figure(4);
[X,Y,Z] = sphere;
axes2 = axes('Parent',figure2,'CameraViewAngle',9.51951638462445,'DataAspectRatio',[1 1 1]);
view(axes2,[124.5 24]);
hold(axes2,'on');
surf(X,Y,Z,'Parent',axes2,'FaceLighting','none','EdgeLighting','flat','FaceColor','none','EdgeColor',[0.8 0.8 0.8]);

%  ToolTipOri_Desired
load('ToolTipOri_Desired.mat');
ToolTipOri_Desired = ActualPos( 2:7 , : );
plot3(ToolTipOri_Desired( 4 , : ),ToolTipOri_Desired( 5 , : ),ToolTipOri_Desired( 6 , : ),'b','Linewidth',2);
hold on
% ToolTipOri_PID
load('ToolTipOri_PID.mat');
ToolTipOri_PID = ActualPos( 2:7 , : );
plot3(ToolTipOri_PID( 4 , : ),ToolTipOri_PID( 5 , : ),ToolTipOri_PID( 6 , : ),'c--','Linewidth',2);
hold on
% ToolTipOri_DNN
plot3(Oa( : , 1 ),Oa( : , 2 ),Oa( : , 3 ),'g','Linewidth',2);                                                                                                                                                                                                                                                           
hold on
% ToolTipOri_DNNPID
load('ToolTipOri_DNNPID.mat');
ToolTipOri_DNNPID = Proposed( 2:7 , : );
plot3(ToolTipOri_DNNPID( 4 , : ),ToolTipOri_DNNPID( 5 , : ),ToolTipOri_DNNPID( 6 , : ),'r--','Linewidth',2);

% legend('Desired', 'PID', 'DNN', 'DNNPID')
zlabel('Ok');
ylabel('Oj');
xlabel('Oi');
title('Tool orientation spline');
set(gca,'FontSize',14);
hold off












