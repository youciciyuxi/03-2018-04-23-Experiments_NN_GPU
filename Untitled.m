



%% Plot the desired trajectory and the predicted trajectory
figure1 = figure(1);
axes1 = axes('Parent',figure1);
view(axes1,[47.5 60]);
hold(axes1,'on');

%  ToolTipOri_Desired
load('ToolTipOri_Desired.mat');
ToolTipOri_Desired = Proposed( 2:7 , : );
plot3(ToolTipOri_Desired( 1 , : ),ToolTipOri_Desired( 2 , : ),ToolTipOri_Desired( 3 , : ),'b','Linewidth',2);
hold on
% ToolTipOri_PID
load('ToolTipOri_PID.mat');
ToolTipOri_PID = Proposed( 2:7 , : );
plot3(ToolTipOri_PID( 1 , : ),ToolTipOri_PID( 2 , : ),ToolTipOri_PID( 3 , : ),'b--','Linewidth',2);
hold on
% ToolTipOri_DNN
load('output_preditionnet.mat');
[Pa,Oa] = ForwardKinematics_DH(output_preditionnet');
plot3(Pa( : , 1 ),Pa( : , 2 ),Pa( : , 3 ),'g','Linewidth',2);
hold on
% ToolTipOri_DNNPID
load('TaskPosition.mat');
ToolTipOri_DNNPID = ActualPos( 2:7 , : );
plot3(ToolTipOri_DNNPID( 1 , : ),ToolTipOri_DNNPID( 2 , : ),ToolTipOri_DNNPID( 3 , : ),'r--','Linewidth',2);

legend('Desired', 'PID', 'DNN', 'DNNPID')
xlabel('X [mm]')
ylabel('Y [mm]')
zlabel('Z [mm]')
title('Tool tip position spline');
set(gca,'FontSize',14);
hold off;

% Orientation
figure2 = figure(2);
[X,Y,Z] = sphere;
axes2 = axes('Parent',figure2,'CameraViewAngle',9.51951638462445,'DataAspectRatio',[1 1 1]);
view(axes2,[124.5 24]);
hold(axes2,'on');
surf(X,Y,Z,'Parent',axes2,'FaceLighting','none','EdgeLighting','flat','FaceColor','none','EdgeColor',[0.8 0.8 0.8]);

%  ToolTipOri_Desired
load('ToolTipOri_Desired.mat');
ToolTipOri_Desired = Proposed( 2:7 , : );
plot3(ToolTipOri_Desired( 4 , : ),ToolTipOri_Desired( 5 , : ),ToolTipOri_Desired( 6 , : ),'b','Linewidth',2);
hold on
% ToolTipOri_PID
load('ToolTipOri_PID.mat');
ToolTipOri_PID = Proposed( 2:7 , : );
plot3(ToolTipOri_PID( 4 , : ),ToolTipOri_PID( 5 , : ),ToolTipOri_PID( 6 , : ),'b--','Linewidth',2);
hold on
% ToolTipOri_DNN
plot3(Oa( : , 1 ),Oa( : , 2 ),Oa( : , 3 ),'g','Linewidth',2);                                                                                                                                                                                                                                                           
hold on
% ToolTipOri_DNNPID
load('TaskPosition.mat');
ToolTipOri_DNNPID = ActualPos( 2:7 , : );
plot3(ToolTipOri_DNNPID( 4 , : ),ToolTipOri_DNNPID( 5 , : ),ToolTipOri_DNNPID( 6 , : ),'r--','Linewidth',2);

% legend('Desired', 'PID', 'DNN', 'DNNPID')
zlabel('Ok');
ylabel('Oj');
xlabel('Oi');
title('Tool orientation spline');
set(gca,'FontSize',14);
hold off