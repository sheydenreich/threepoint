<!-- PROJECT LOGO -->

<br />

<h3 align="center">threepoint </h3>

<p align="center">
    Code for modelling the third-order aperture statistics and the shear three-point correlation function  <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This code models the third-order aperture statistics $\langle M_\mathrm{ap}^3 \rangle$ and the natural components of the shear three-point correlation function $\Gamma_i$ a. The model is based on the `BiHalofit`-Bispectrum model by [Takahashi+ (2021)](https://ui.adsabs.harvard.edu/abs/2020ApJ...895..113T/abstract). The modelling of the $\Gamma_i$ uses the integration routine by [Ogata+ (2005)](https://www.kurims.kyoto-u.ac.jp/~prims/pdf/41-4/41-4-40.pdf)

The modelling of $\langle M_\mathrm{ap}^3 \rangle$ and $\Gamma_i$ is described in detail in Heydenreich+ (2022).  Please cite this publication if you use this code in your project.

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

To use this code these requirements are needed:

* **Cuba** (Tested for version 4.2.2). Check here for how to install it: [http://www.feynarts.de/cuba/](http://www.feynarts.de/cuba/)
* **GNU Scientific Library** (Tested for version 2.6). Check here for how to install it: [https://www.gnu.org/software/gsl/](https://www.gnu.org/software/gsl/)
* **cosmosis** Check here for how to install it: [https://cosmosis.readthedocs.io/en/latest/](https://cosmosis.readthedocs.io/en/latest/)
* **g++** (Tested for version 9.3.0).
  Under Ubuntu this can be installed with

  ```sh
  sudo apt install build-essential
  ```
* **openMP** (Tested for version 4.5). Under Ubuntu this can be installed with

```sh
sudo apt-get install libomp-dev
```

At the moment, only the GPU accelerated version is fully tested and released (although a CPU-only version might be available in the future). Therefore  additionally the following is needed:

* **NVIDIA graphics card with CUDA capability of at least 2**. Check here to see, if your card works: [https://en.wikipedia.org/wiki/CUDA#GPUs_supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
* **CUDA SDK Toolkit** (Tested for version 10.1, at least 7 needed!)
  Can be downloaded here [https://developer.nvidia.com/accelerated-computing-toolkit](https://developer.nvidia.com/accelerated-computing-toolkit)

In general, some knowledge on CUDA and how GPUs work is useful to understand the code!

### Installation

1. Download the newest release and unpack it
2. Install missing prerequisites
3. Go into the source directory and open Makefile

```sh
cd cuda_version
xdg-open Makefile
```

4. Under `LFLAGS` add the path to your instance of `lcosmosis` and `lcuba`
5. Under `INCLUDES` add the path to the source code for the cosmosis datablock
6. run `make`
7. Now, check if the folder `cuda_version` contains the necessary executables.

<!-- USAGE EXAMPLES -->

## Usage

The folder `examples` contains examples for the required parameter files and how the code can be run. Three quantities are the key outputs of the code: the aperture statistics $\langle M_\mathrm{ap}^3\rangle$, the natural components of the 3Pt-Correlation Function $\Gamma$ and the covariance $C_{M^3}$ of the aperture statistics.

Note that the number of redshift bins used for the interpolation of the lensing weights (and other quantities) is hard-coded as `n_redshift_bins` in the file `bispectrum.cuh`. Adjust this if necessary.

### Calculation of Third order aperture statistics

#### Required Input

The calculation requires the following input:

* A parameter file containing the cosmological parameters (see `exampleCosmology.param` for an example)
* A parameter file containing the aperture radii for which the statistics shall be calculated in arcmin (see `exampleThetas.dat` for an example)
* The redshift distribution of source galaxies (see `exampleNz.dat` for an example)

#### Output

The code writes the $\langle M_\mathrm{ap}^3\rangle$ into the file specified in the function call. This file will be an ASCII file with the columns:

1. Aperture scale radius 1 [arcmin]
2. Aperture scale radius 2 [arcmin]
3. Aperture scale radius 3 [arcmin]
4. $\langle M_\mathrm{ap}^3\rangle$

### Calculation of 3Pt-Correlation Function

#### Required Input

The calculation requires the following input:

* A parameter file containing the cosmological parameters (see `exampleCosmology.param` for an example)
* A parameter file containing the triangle configurations for which the statistics shall be calculated in arcmin (see `exampleGammaconfig.dat` for an example).
* The redshift distribution of source galaxies (see `exampleNz.dat` for an example)

#### Output

The code writes the $\Gamma_i$ into the file specified in the function call. This file will be an ASCII file with the columns:

1. Real part of $\Gamma_0$
2. Imaginary part of $\Gamma_0$
3. Real part of $\Gamma_1$
4. Imaginary part of $\Gamma_1$
5. Real part of $\Gamma_2$
6. Imaginary part of $\Gamma_2$
7. Real part of $\Gamma_3$
8. Imaginary part of $\Gamma_3$
9. $r$-bin
10. $u$-bin
11. $v$-bin

$r, u,$ and $v$ are the triangle parameters defined by Jarvis+(2004) and implemented in `treecorr`. See here for their definition: [https://rmjarvis.github.io/TreeCorr/_build/html/correlation3.html](https://rmjarvis.github.io/TreeCorr/_build/html/correlation3.html).
By default, the triangle center is set as the centroid, to ensure compatibility with the definitions in `treecorr` (Jarvis+2004). This can be deactivated when disabling the flag `CONVERT_TO_CENTROID` in the file `gamma.cuh`.

<!-- LICENSE -->

## License

Distributed under the GNU General Public License v 3.0

If you use the code for a publication, please cite Heydenreich+ (2022).

<!-- CONTACT -->

## Contact

Sven Heydenreich - [sven@astro.uni-bonn.de](mailto:sven@astro.uni-bonn.de)
Laila Linke -  [llinke@astro.uni-bonn.de](mailto:llinke@astro.uni-bonn.de)

Project Link: [https://github.com/sheydenreich/threepoint](https://github.com/sheydenreich/threepoint)

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* The integration routines use [cubature](https://github.com/stevengj/cubature) and [Cuba](http://www.feynarts.de/cuba/)
* This ReadMe is based on [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
