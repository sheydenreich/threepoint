#include <iostream>
#include <vector>
#include <complex>


double U(double vartheta, double theta)
{
    double x=vartheta/theta;
    x=x*x/2;
    return 1/(2*M_PI*theta*theta)*exp(-x)*(1.0-x);
}

void getVarthetas(int Npix, double deltaTheta, std::vector<std::complex<double>>& varthetas)
{
#pragma omp parallel for
    for(int i=0; i<Npix; i++)
    {
        double x=i*deltaTheta;

        for(int j=0; j<Npix; j++)
        {
            double y=j*deltaTheta;
            std::complex<double> coord(x,y);
            varthetas.push_back(coord);
        }
    }
}

void getAs(double theta1, double theta2, double theta3, 
            double theta4, double theta5, double theta6,
            int Npix, const std::vector<std::complex<double>>& varthetas,
            std::vector<double>& A12, std::vector<double>& A13, std::vector<double>& A14, std::vector<double>& A15,
            std::vector<double>& A16, std::vector<double>& A23, std::vector<double>& A24, std::vector<double>& A25,
            std::vector<double>& A26, std::vector<double>& A34, std::vector<double>& A35, std::vector<double>& A36,            
            std::vector<double>& A45, std::vector<double>& A46, std::vector<double>& A56)
{
    int Npix2d=Npix*Npix;
#pragma omp parallel for
    for(int i=0; i<Npix2d; i++)
    {
        std::complex<double> vartheta_i=varthetas.at(i);
        for(int j=0; j<Npix2d; j++)
        {
            double a12, a13, a14, a15, a16, a23, a24, a25, a26, a34, a35, a36, a45, a46, a56;
            a12=a13=a14=a15=a16=a23=a24=a25=a26=a34=a35=a36=a45=a46=a56;
            std::complex<double> vartheta_j=varthetas.at(j);
            for(int k=0; k<Npix2d; k++)
            {
                std::cerr<<i<<" "<<j<<" "<<k<<'\r';
	            std::cerr.flush();
                std::complex<double> vartheta_k=varthetas.at(k);
                double distance_1=std::abs(vartheta_i-vartheta_k);
                double distance_2=std::abs(vartheta_j-vartheta_k);
                a12+=U(distance_1, theta1)*U(distance_2,theta2);
                a13+=U(distance_1, theta1)*U(distance_2,theta3);
                a14+=U(distance_1, theta1)*U(distance_2,theta4);
                a15+=U(distance_1, theta1)*U(distance_2,theta5);
                a16+=U(distance_1, theta1)*U(distance_2,theta6);
                a23+=U(distance_1, theta2)*U(distance_2,theta3);
                a24+=U(distance_1, theta2)*U(distance_2,theta4);
                a25+=U(distance_1, theta2)*U(distance_2,theta5);
                a26+=U(distance_1, theta2)*U(distance_2,theta6);
                a34+=U(distance_1, theta3)*U(distance_2,theta4);
                a35+=U(distance_1, theta3)*U(distance_2,theta5);
                a36+=U(distance_1, theta3)*U(distance_2,theta6);
                a45+=U(distance_1, theta4)*U(distance_2,theta5);
                a46+=U(distance_1, theta4)*U(distance_2,theta6);
            }
            A12.push_back(a12);
            A13.push_back(a13);
            A14.push_back(a14);
            A15.push_back(a15);            
            A16.push_back(a16);
            A23.push_back(a23);
            A24.push_back(a24);
            A25.push_back(a25);
            A26.push_back(a26);
            A34.push_back(a34);
            A35.push_back(a35);
            A36.push_back(a36);
            A45.push_back(a45);
            A46.push_back(a46);
            A56.push_back(a56);
        }
    }
}

double term1(int Npix2d, std::vector<double>& A12, std::vector<double>& A13, std::vector<double>& A14, std::vector<double>& A15,
            std::vector<double>& A16, std::vector<double>& A23, std::vector<double>& A24, std::vector<double>& A25,
            std::vector<double>& A26, std::vector<double>& A34, std::vector<double>& A35, std::vector<double>& A36,            
            std::vector<double>& A45, std::vector<double>& A46, std::vector<double>& A56)
{
    double t1=0;
#pragma omp parallel for
for(int i=0; i<Npix2d; i++)
{
    for(int j=0; j<Npix2d; j++)
    {
        t1+=A14[i*Npix2d+j]*A25[i*Npix2d+j]*A36[i*Npix2d+j];
        t1+=A14[i*Npix2d+j]*A26[i*Npix2d+j]*A35[i*Npix2d+j];
        t1+=A15[i*Npix2d+j]*A26[i*Npix2d+j]*A34[i*Npix2d+j];
        t1+=A15[i*Npix2d+j]*A24[i*Npix2d+j]*A36[i*Npix2d+j];
        t1+=A16[i*Npix2d+j]*A24[i*Npix2d+j]*A35[i*Npix2d+j];
        t1+=A16[i*Npix2d+j]*A25[i*Npix2d+j]*A34[i*Npix2d+j];            
    }
}
return t1;
}

double term2(int Npix2d, std::vector<double>& A12, std::vector<double>& A13, std::vector<double>& A14, std::vector<double>& A15,
            std::vector<double>& A16, std::vector<double>& A23, std::vector<double>& A24, std::vector<double>& A25,
            std::vector<double>& A26, std::vector<double>& A34, std::vector<double>& A35, std::vector<double>& A36,            
            std::vector<double>& A45, std::vector<double>& A46, std::vector<double>& A56)
{
double t2=0;
#pragma omp parallel for
for(int i=0; i<Npix2d; i++)
{
    for(int j=0; j<Npix2d; j++)
    {
        t2+=A12[i*Npix2d+i]*A34[i*Npix2d+j]*A56[j*Npix2d+j];
        t2+=A12[i*Npix2d+i]*A35[i*Npix2d+j]*A46[j*Npix2d+j];
        t2+=A12[i*Npix2d+i]*A36[i*Npix2d+j]*A45[j*Npix2d+j];
        t2+=A13[i*Npix2d+i]*A24[i*Npix2d+j]*A56[j*Npix2d+j];
        t2+=A13[i*Npix2d+i]*A25[i*Npix2d+j]*A46[j*Npix2d+j];
        t2+=A13[i*Npix2d+i]*A26[i*Npix2d+j]*A45[j*Npix2d+j];
        t2+=A23[i*Npix2d+i]*A14[i*Npix2d+j]*A56[j*Npix2d+j];
        t2+=A23[i*Npix2d+i]*A15[i*Npix2d+j]*A46[j*Npix2d+j];
        t2+=A23[i*Npix2d+i]*A16[i*Npix2d+j]*A45[j*Npix2d+j];
    }
}
return t2;
}

void totalCov(double theta1, double theta2, double theta3, double theta4, double theta5, double theta6,
                int Npix, double Apix, double sigma, double deltaTheta, const std::vector<std::complex<double>>& varthetas,
                double& t1, double& t2, double& total)
{
    int Npix2d=Npix*Npix;
    double norm=pow(Apix*sigma, 6)/Npix/Npix;
    std::vector<double> A12,  A13,  A14,  A15, A16,  A23,  A24,  A25, A26,  A34,  A35,  A36, A45,  A46,  A56;

    getAs(theta1, theta2, theta3, theta4, theta5, theta6, Npix, varthetas, A12, A13, A14, A15, 
    A16,  A23,  A24,  A25, A26,  A34,  A35,  A36, A45,  A46,  A56);

    t1=term1(Npix2d, A12,A13, A14, A15, 
                    A16,  A23,  A24,  A25, A26,  A34,  A35,  A36, A45,  A46,  A56);
    t1*=norm;
    std::cerr<<"Term1:"<<t1<<std::endl;

    t2=term2(Npix2d, A12,A13, A14, A15, 
                    A16,  A23,  A24,  A25, A26,  A34,  A35,  A36, A45,  A46,  A56);
    t2*=norm;
    std::cerr<<"Term2:"<<t2<<std::endl;

    std::cerr<<"Term1:"<<t1<<" "<<"Term2:"<<t2<<std::endl;
    total=t1+t2;
    std::cerr<<"Total:"<<total<<std::endl;
}


int main(int argc, char* argv[])
{
    int Npix=std::stoi(argv[1]);
    double side=std::stod(argv[2])*M_PI/180;

    double A=side*side;
    double Apix=A/Npix/Npix;
    double deltaTheta=side/Npix;
    double sigma=0.3;

    std::vector<double> thetas{2,8};//,4,8,16};
    int N=thetas.size();

    std::vector<std::complex<double>> varthetas;
    getVarthetas(Npix, deltaTheta, varthetas);
    std::cerr<<"Finished creating varthetas"<<std::endl;

    for(int i=0; i<N; i++)
    {
        double theta1=thetas.at(i)*M_PI/180./60.;
        for(int j=i; j<N; j++)
        {
            double theta2=thetas.at(j)*M_PI/180./60.;
            for(int k=j; k<N; k++)
            {
                double theta3=thetas.at(k)*M_PI/180./60.;
                for(int l=i; l<N; l++)
                {
                    double theta4=thetas.at(l)*M_PI/180./60.;
                    for(int m=l; m<N; m++)
                    {
                        double theta5=thetas.at(m)*M_PI/180./60.;
                        for(int n=m; n<N; n++)
                        {
                            double theta6=thetas.at(n)*M_PI/180./60.;
                            double t1, t2, total;
                            totalCov(theta1, theta2, theta3, theta4, theta5, theta6, Npix, Apix, sigma, deltaTheta, varthetas, t1, t2, total);
                            std::cout<<theta1<<" "
                                    <<theta2<<" "
                                    <<theta3<<" "
                                    <<theta4<<" "
                                    <<theta5<<" "
                                    <<theta6<<" "
                                    <<t1<<" "<<t2<<" "<<total<<std::endl;

                        }
                    }
                }
            }
        }
    }


    return 0;

}