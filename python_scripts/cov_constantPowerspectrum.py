import numpy as np


class cov_constantPowerspectrum:
    """ Class for calculating the <Map3> covariance for a field containing only shape noise

    Based on my calculations at https://1drv.ms/b/s!AhKG5KH8zyk3hed0Ge59VxLfefc2KQ?e=y279UE
    The class calculates the covariance for given shape noise, source density and survey area,
    assuming that there is no shear signal. It is useful for tests of the covariance estimation

    Attributes
    ----------
    sigma : float
        Shapenoise, usually ~ 0.3
    n : float
        source galaxy number density [rad^-2]
    A : float
        survey area [rad^2]
    """

    def __init__(self, sigma, n, A, unit="rad"):
        """ Class constructor

        Sets the attributes sigma, n, and A (and converts them to radian-units)

        Parameters
        ----------
        sigma : float
            Shapenoise
        n : float
            source galaxy number density [unit^-2]
        A : float
            survey area [unit^2]
        unit : string (optional)
            angular unit of n and A, can be "rad", "arcmin" or "deg" (default : rad)
        """

        # Set shapenoise
        self.sigma=sigma

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

        # Set n and A
        self.n=n/conversion/conversion

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

        a=0.5*(th1sq+th3sq+th4sq+th6sq)
        b=0.5*(th2sq+th3sq+th5sq+th6sq)
        c=(th3sq+th6sq)

        d=pow(4*a*b-c*c, 7)

        G=64*a*a*b*b*(2*a*a+3*a*b+2*b*b)+16*a*b*c*c*(16*a*a+27*a*b+16*b*b)+12*pow(c,4)*(4*a*a+9*a*b+4*b*b)+3*pow(c, 6)
        G*=1536/d


        H=192*pow(a*b,3)-768*a*a*b*b*c*(a+b)+48*pow(a*b*c,2)-576*a*b*pow(c,3)*(a+b)-20*a*b*pow(c,4)-48*(a+b)*pow(c,5)-pow(c,6)
        H*=384/d

        
        I=96*a*a*b*b+32*a*b*c*c+pow(c,4)
        I*=3072*c*c/d

        
        return G+4*H+4*I


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
        N=pow(self.sigma, 6)/pow(2*self.n, 3)/64*pow(thetas[0]*thetas[1]*thetas[2]*thetas[3]*thetas[4]*thetas[5], 2)/self.A/pow(2*np.pi, 3)
        cov*=2*np.pi*N
        return cov
