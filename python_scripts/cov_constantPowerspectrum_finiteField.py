import numpy as np
from scipy.special import erf


class cov_constantPowerspectrum_finiteField:
    """ Class for calculating the <Map3> covariance for a square field containing only shape noise for a finite field size

    Based on Eq. 96 in my calculations at https://1drv.ms/b/s!AhKG5KH8zyk3hed0Ge59VxLfefc2KQ?e=y279UE
    The class calculates the covariance for given shape noise, source density and survey area,
    assuming that there is no shear signal. It is useful for tests of the covariance estimation

    Attributes
    ----------
    sigma : float
        Shapenoise, usually ~ 0.3
    n : float
        source galaxy number density [rad^-2]
    a : float
        side length of field [rad]
    Pcubed : float
        Powerspectrum, i.e. sigma^6/n^3
    """

    def __init__(self, sigma, n, A, unit="rad"):
        """ Class constructor

        Sets the attributes sigma, n, and a (and converts them to radian-units)

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

        # Set n and a
        self.n=n/conversion/conversion

        self.a=np.sqrt(A)*conversion

        self.Pcubed=self.sigma**6/self.n**3




    def term1(self, theta1, theta2, theta3, theta4, theta5, theta6):
        """Calculates one permutation of the first term (without multiplication by sigma^6/n^3), i.e. L1 in the document

        Args:
            theta1 (float): Aperture Radius 1 [rad]
            theta2 (float): Aperture Radius 2 [rad]
            theta3 (float): Aperture Radius 3 [rad]
            theta4 (float): Aperture Radius 4 [rad]
            theta5 (float): Aperture Radius 5 [rad]
            theta6 (float): Aperture Radius 6 [rad]

        Returns:
            float: 1 Permutation of Term 1 (Also called L1 in the document)
        """
        
        a=self.a

        b1=theta1**2+theta4**2
        b2=theta2**2+theta5**2
        b3=theta3**2+theta6**2

        b=1/b1+1/b2+1/b3

        N=theta1*theta2*theta3*theta4*theta5*theta6
        N=N*N/a**4/np.pi**3/b1**5/b2**5/b3**5

        A=((b1*b2*b3)**2)
        B = (-b1*b2**2*b3**2 - b2*b1**2*b3**2 - b3*b1**2*b2**2)
        C = (b1*b2*b3**2 + b1*b3*b2**2 +  b2*b3*b1**2 + (b1**2*b2**2 + b1**2*b3**2 + b1**2*b3**2)/8)
        D = (-(b2*b3**2 + b3*b2**2 + b1*b3**2 + b3*b1**2 + b1*b2**2 + b2*b1**2)/8 - b1*b2*b3)
        E = ((b1*b2 + b1*b3 + b2*b3)/8 + (b3**2 + b2**2 + b1**2)/64)
        F = -(b1 + b2 + b3)/64
        G = 1.0/512

        h=a*np.sqrt(b)*np.sqrt(2*np.pi)*erf(a*np.sqrt(b/2))
        k=np.exp(-0.5*a**2*b)
        l=a**2*b

        T0=(2*k+h-2)/b
        T1=(4*k+h-4)/b**2
        T2=(2*k*(l+8)+3*h-16)/b**3
        T3=(k*(2*l*(l+9)+96)+15*h-96)/b**4
        T4=(k*(2*l*(l*(l+13)+87)+768)+105*h-768)/b**5
        T5=(k*(2*l*(l*(l*(l+17)+165)+975)+7680)+945*h-7680)/b**6
        T6=(k*(2*l*(l*(l*(l*(l+21)+267)+2295)+12645)+92160)+10395*h-92160)/b**7


        C0=T0**2
        C1=2*T0*T1
        C2=2*(T0*T2+T1**2)
        C3=2*(T0*T3+3*T2*T1)
        C4=2*(T0*T4+4*T1*T3+3*T2**2)
        C5=2*(T0*T5+5*T1*T4+10*T2*T3)
        C6=2*(T0*T6+6*T1*T5+14*T2*T4+10*T3**2)

        result=A*C0+B*C1+C*C2+D*C3+E*C4+F*C5+G*C6

        result*=N


        return result


    def term2(self, theta1, theta2, theta3, theta4, theta5, theta6):
        """Calculates one permutation of the second term (without multiplication by sigma^6/n^3), i.e. L2 in the document

        Args:
            theta1 (float): Aperture Radius 1 [rad]
            theta2 (float): Aperture Radius 2 [rad]
            theta3 (float): Aperture Radius 3 [rad]
            theta4 (float): Aperture Radius 4 [rad]
            theta5 (float): Aperture Radius 5 [rad]
            theta6 (float): Aperture Radius 6 [rad]

        Returns:
            float: 1 Permutation of Term 2 (Also called L2 in the document)
        """
        a=self.a

        K1=(theta1*theta2)**2/np.pi/(theta1**2+theta2**2)**3
        K2=(theta5*theta6)**2/np.pi/(theta5**2+theta6**2)**3

        b=theta3**2+theta4**2
        N=theta3**2*theta4**2/np.pi/b**5

        result=np.sqrt(np.pi/2)*pow(b, 5/2)/a**3*erf(a/np.sqrt(2*b))*(b*(1-np.exp(-a*a/2/b))+a*a*np.exp(-a*a/2/b))
        result-=b**3/a**2*(np.exp(-a**2/2/b)-np.exp(-a**2/b))

        result*=N*K1*K2
        return result



    def term1_total(self, thetas, unit='rad'):
        """Calculates all Permutations of L1

        Args:
            thetas (np array, containing 6 values): Apertureradii [unit]
            unit (str, optional): Unit of aperture radii. Possible values: "rad", "arcmin", and "deg". Defaults to 'rad'.

        Returns:
            float: All Permutations of L1, multiplied with P(\ell)^3
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

        # Do calculation
        T1=self.term1(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5])
        T1+=self.term1(thetas[0], thetas[1], thetas[2], thetas[3], thetas[5], thetas[4])
        T1+=self.term1(thetas[0], thetas[1], thetas[2], thetas[4], thetas[3], thetas[5])
        T1+=self.term1(thetas[0], thetas[1], thetas[2], thetas[4], thetas[5], thetas[3])
        T1+=self.term1(thetas[0], thetas[1], thetas[2], thetas[5], thetas[3], thetas[4])
        T1+=self.term1(thetas[0], thetas[1], thetas[2], thetas[5], thetas[4], thetas[3])


        return T1*self.Pcubed/2/np.pi

    def term2_total(self, thetas, unit='rad'):
        """Calculates all Permutations of L2

        Args:
            thetas (np array, containing 6 values): Apertureradii [unit]
            unit (str, optional): Unit of aperture radii. Possible values: "rad", "arcmin", and "deg". Defaults to 'rad'.

        Returns:
            float: All Permutations of L2, multiplied with P(\ell)^3
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

   
        T2=self.term2(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4], thetas[5])
        T2+=self.term2(thetas[0], thetas[1], thetas[2], thetas[4], thetas[3], thetas[5])
        T2+=self.term2(thetas[0], thetas[1], thetas[2], thetas[5], thetas[3], thetas[4])
        T2+=self.term2(thetas[0], thetas[2], thetas[1], thetas[3], thetas[4], thetas[5])
        T2+=self.term2(thetas[0], thetas[2], thetas[1], thetas[4], thetas[3], thetas[5])
        T2+=self.term2(thetas[0], thetas[2], thetas[1], thetas[5], thetas[3], thetas[4])
        T2+=self.term2(thetas[2], thetas[1], thetas[0], thetas[3], thetas[4], thetas[5])
        T2+=self.term2(thetas[2], thetas[1], thetas[0], thetas[4], thetas[3], thetas[5])
        T2+=self.term2(thetas[2], thetas[1], thetas[0], thetas[5], thetas[3], thetas[4])
        return T2*self.Pcubed/2/np.pi


    def covariance(self, thetas, unit='rad'):
        """Calculates total covariance, i.e. L1+L2+Perm

        Args:
            thetas (np array, containing 6 values): Apertureradii [unit]
            unit (str, optional): Unit of aperture radii. Possible values: "rad", "arcmin", and "deg". Defaults to 'rad'.

        Returns:
            float: Covariance at thetas
        """
        T1=self.term1_total(thetas, unit)
        T2=self.term2_total(thetas, unit)

        return T1+T2