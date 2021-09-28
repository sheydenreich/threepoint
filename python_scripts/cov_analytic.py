import numpy as np

class cov_analytic:
    """ Class for calculating the <Map3> covariance for a Gaussian field with Powerspectrum P(l)=p1*l^2*exp(-p2*l^2)

    The class calculates the covariance for given parameters p1 and p2 of the Powerspectrum and survey area. It is useful for tests of the covariance estimation

    Attributes
    ----------
    p1 : float
        Parameter 1 of Powerspectrum [rad]
    p2 : float
        Parameter 2 of Powerspectrum [rad]
    A : float
        Survey area [rad^2]
    """

    def __init__(self, p1, p2, A, unit="rad"):
        """ Class constructor
        
        Sets the attributes sigma, n, and A (and converts them to radian-units)
        
        Parameters
        ----------
        p1 : float
            Parameter 1 of Powerspectrum [rad]
        p2 : float
            Parameter 2 of Powerspectrum [rad]
        A : float
            survey area [unit^2]
        unit : string (optional)
            angular unit of A, can be "rad", "arcmin" or "deg" (default : rad)
        """

        # Set parameter of Powerspectrum
        self.p1=p1
        self.p2=p2

        # Determine conversion from "unit" to "rad"
        # (In python 3.10 this could be a match/case statement, but this is not released yet)
        if(unit=="rad"):
            conversion=1
        elif(unit=="arcmin"):
            conversion=np.pi/180./60.
        elif(unit=="deg"):
            conversion=np.pi/180.
        else:
            print("Invalid unit given for initialization of cov_random. Aborting.")
            exit()

        # Set A
        self.A=A*conversion*conversion

    def covariance_1perm(self, theta1, theta2, theta3, theta4, theta5, theta6):
        """ Calculates one permutation of the covariance (wo normalization)

        Parameters
        ----------
        theta1 : float
            First Apertureradius [rad]
        theta2 : float
            Second Apertureradius [rad]
        theta3 : float
            Third Apertureradius [rad]
        theta4 : float
            Fourth Apertureradius [rad]
        theta5 : float
            Fifth Apertureradius [rad]
        theta6 : float
            Sixth Apertureradius [rad]
        """

        # Helper variables
        th1sq=theta1*theta1
        th2sq=theta2*theta2
        th3sq=theta3*theta3
        th4sq=theta4*theta4
        th5sq=theta5*theta5
        th6sq=theta6*theta6

        a=-0.5*(th1sq+th3sq+th4sq+th6sq)-2*self.p2
        b=-0.5*(th2sq+th3sq+th5sq+th6sq)-2*self.p2
        c=-th3sq-th6sq-2*self.p2

        d=pow(c*c-4*a*b, 10)

        G=128*pow(a*b, 3)*(a*a+a*b+b*b)+64*pow(a*b*c,2)*(9*a*a+11*a*b+9*b*b)+120*a*b*pow(c,4)*(3*a*a+5*a*b+3*b*b)+40*pow(c,6)*(a*a+3*a*b+b*b)+5*pow(c,8)
        G*=-8847360*(a+b)/d

        H=-256*pow(a*b,4)*(a+b)+256*pow(a*b,3)*(3*a*a+5*a*b+3*b*b)*c-320*pow(a*b,3)*(a+b)*c*c+160*pow(a*b,2)*(9*a*a+16*a*b+9*b*b)*pow(c,3)+480*a*b*pow(a+b,2)*pow(c,5)+20*a*b*(a+b)*pow(c,6)+10*(3*a*a+8*a*b+3*b*b)*pow(c,7)+(a+b)*pow(c, 8)+pow(c,9)
        H*=17694720/d

        J=-2560*pow(a*b,3)*a*(-b+3*c)-pow(c,7)*(48*b+c)-1280*pow(a*b,2)*a*c*(6*b*b-b*c+6*c*c)-240*pow(a*c,2)*b*c*(32*b*b+b*c+6*c*c)-8*a*pow(c,5)*(180*b*b+7*b*c+6*c*c)
        J*=4423680*c/d

        K=1280*pow(a*b, 3)+720*pow(a*b*c,2)+72*a*b*pow(c,4)+pow(c,6)
        K*=17694720*pow(c,3)/d

        return G+H+J+K




    def covariance(self, thetas, unit="rad"):
        """ Calculates the covariance

        Parameters
        ----------
        thetas : np.array (should have length 6!)
            Apertureradii [unit]
        unit : string (optional)
            angular unit of aperture radii, can be "rad", "arcmin", or "deg" (default : rad)
        """

        # Check if aperture radii have correct dimensions
        if(len(thetas)!=6):
            print("Not the right number of aperture radii in cov_random.covariance. Aborting.")
            exit()
        
        
        # Determine conversion from "unit" to "rad"
        # (In python 3.10 this could be a match/case statement, but this is not released yet)
        if(unit== "rad"):
            conversion=1
        elif(unit=="arcmin"):
            conversion=np.pi/180./60.
        elif(unit=="deg"):
            conversion=np.pi/180.
        else:
            print("Invalid unit given for cov_random.covariance. Aborting.")
            exit()


        # Conversion
        thetas=thetas*conversion

        # Calculation + permutations
        cov=self.covariance_1perm(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5])
        cov+=self.covariance_1perm(thetas[0], thetas[1], thetas[2], thetas[3], thetas[5], thetas[4])
        cov+=self.covariance_1perm(thetas[0], thetas[1], thetas[2], thetas[4], thetas[3], thetas[5])
        cov+=self.covariance_1perm(thetas[0], thetas[1], thetas[2], thetas[4], thetas[5], thetas[3])
        cov+=self.covariance_1perm(thetas[0], thetas[1], thetas[2], thetas[5], thetas[3], thetas[4])
        cov+=self.covariance_1perm(thetas[0], thetas[1], thetas[2], thetas[5], thetas[4], thetas[3])

        # Normalisation
        N=pow(self.p1/2/np.pi, 3)/self.A/64*pow(thetas[0]*thetas[1]*thetas[2]*thetas[3]*thetas[4]*thetas[5], 2)
        cov*=2*np.pi*N
        return cov
