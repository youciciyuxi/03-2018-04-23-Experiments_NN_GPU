function [ Curvature ] = CurvatureCal( DeBoorPointDer012 )

sizeDeBoorP=size(DeBoorPointDer012);
Curvature=zeros(sizeDeBoorP(3),1);
for ii=1:sizeDeBoorP(3)
    % ��ȡ���������
    xKnotCorDer1=DeBoorPointDer012(2,1,ii);
    yKnotCorDer1=DeBoorPointDer012(2,2,ii);
    zKnotCorDer1=DeBoorPointDer012(2,3,ii);
    xKnotCorDer2=DeBoorPointDer012(3,1,ii);
    yKnotCorDer2=DeBoorPointDer012(3,2,ii);
    zKnotCorDer2=DeBoorPointDer012(3,3,ii);

    % �ټ��㸱��ʸ���ķ���
    subNormalVectorX1=yKnotCorDer1*zKnotCorDer2-yKnotCorDer2*zKnotCorDer1;
    subNormalVectorY1=-xKnotCorDer1*zKnotCorDer2+xKnotCorDer2*zKnotCorDer1;
    subNormalVectorZ1=xKnotCorDer1*yKnotCorDer2-xKnotCorDer2*yKnotCorDer1;

    % ��������岹�������
    CurvatureNum=sqrt(subNormalVectorX1^2+subNormalVectorY1^2+subNormalVectorZ1^2);
    CurvatureDen=(sqrt(xKnotCorDer1^2+yKnotCorDer1^2+zKnotCorDer1^2))^3;
    Curvature(ii)=CurvatureNum/CurvatureDen;
end
end

