
#include "math.h" 
#include "stdlib.h"

/******************************/
// ��������NURBS���߱��ĳ���
/******************************/
// �����ڵ�ʸ����ά��
#define KNOTVECTORLEN 29        // Fan ����
// �������Ƶ�ĸ���
#define CONTROLPOINTQ 25	    // Fan ����
// ����NURBS���ߵĽ���
#define NURBSORDER 3	       // Fan ����


// #define KNOTVECTORLEN 56        // Blade ����
// #define CONTROLPOINTQ 52	    // Blade ����
// #define NURBSORDER 3	       // Blade ����


/******************************/
// ��������de-Boor Cox��ʽ������ֵ��ʱ�õ���ȫ�ֱ���
/******************************/
// ��������õ��ĵ������ֵ�㣬��Cx��Cy��Cz
extern double xKnotCoor;
extern double yKnotCoor;
extern double zKnotCoor;
// ��������õ��ĵ������ֵ���һ�׵�ʸ����Cx'��Cy'��Cz'
extern double xKnotCoorDer1;
extern double yKnotCoorDer1;
extern double zKnotCoorDer1;
// ��������õ��ĵ������ֵ��Ķ��׵�ʸ����Cx"��Cy"��Cz"
extern double xKnotCoorDer2;
extern double yKnotCoorDer2;
extern double zKnotCoorDer2;
// ��������õ��ĵ������ֵ������׵�ʸ����Cx"'��Cy"'��Cz"'
extern double xKnotCoorDer3;
extern double yKnotCoorDer3;
extern double zKnotCoorDer3;
// ��������õ��ĵ��᷽����ֵ�㣬��Ci��Cj��Ck
extern double iKnotCoor;
extern double jKnotCoor;
extern double kKnotCoor;
// ����xyzijk�Ŀ��Ƶ㣬Ȩֵ�Լ��ڵ�ʸ��
extern double knotVector[KNOTVECTORLEN];
extern double xCtrlCoor[CONTROLPOINTQ];
extern double yCtrlCoor[CONTROLPOINTQ];
extern double zCtrlCoor[CONTROLPOINTQ];
extern double iCtrlCoor[CONTROLPOINTQ];
extern double jCtrlCoor[CONTROLPOINTQ];
extern double kCtrlCoor[CONTROLPOINTQ];
extern double weightVector[CONTROLPOINTQ];
// ����w_i*P_i����ֵ����
extern 	double xMulWeight[CONTROLPOINTQ];
extern 	double yMulWeight[CONTROLPOINTQ];
extern 	double zMulWeight[CONTROLPOINTQ];
extern 	double iMulWeight[CONTROLPOINTQ];
extern 	double jMulWeight[CONTROLPOINTQ];
extern 	double kMulWeight[CONTROLPOINTQ];

// ����de-Boor Cox��ʽ�������ֵ���Լ���ֵ��ĵ�ʸ
double *DeboorToolTipOrien(double uParameter);	// ���ڼ�����ֵ�㣬һ���׵�ʸ�����ʺ����ʰ뾶
double DeBoorCoxCal(double alfaMatrix[NURBSORDER][NURBSORDER], double *ctrlPointCor, int nurbsOrder, int uIndex);
double DeBoorCoxDer1Cal(double *knotVector, double alfaMatrix[NURBSORDER][NURBSORDER], double *ctrlPointCor, int nurbsOrder, int uIndex);
double DeBoorCoxDer2Cal(double *knotVector, double alfaMatrix[NURBSORDER][NURBSORDER], double *ctrlPointCor, int nurbsOrder, int uIndex);
double DeBoorCoxDer3Cal(double *knotVector, double *ctrlPointCor, int nurbsOrder, int uIndex);
// �ڼ���һ���׵�ʸʱ����Ҫ����һ���м����ʽ
double TempIterative(double *knotVector, double *ctrlPointCor, int nurbsOrder, int indexInterative);








