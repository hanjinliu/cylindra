# CFT

CFT (cylindric Fourier Transformation) is a method introduced in our work to measure
the local and global lattice parameters of microtubules. It composed of following steps:

1. Coordinate transformation from the Cartesian coordinate $(z, y, x)$ to the cylindrical coordinate $(r, y, \theta)$.
2. 3D Discrete Fourier Transformation around the peak locations with up-sampling. The
   peak locations are defined in the spline configurations.
3. Project the 3D cylindric power spectrum to $(y, \theta)$. The peak frequencies are
   used to calculate the lattice parameters.

??? info "Why not FFT?"
    The FFT (Fast Fourier Transformation) is a widely used algorithm to calculate the
    Fourier Transformation very efficiently. However, FFT is not suitable for
    up-sampling because it results in a very large output image (10&times; up-sampling
    in two axes results in a 100&times; larger image). In this case, discrete Fourier
    transformation is must faster and more memory efficient.

## Estimate the Radius

API: [`measure_radius`][cylindra.widgets.main.CylindraMainWidget.measure_radius]

To run CFT, we have to specify the region to be transformed into the cylindric
coordinate. `Analysis > Radius > Measure radius` is the simplest method to estimate
the radius using the radial profile of sampled sub-volumes along the spline.

## Running CFT

### local-CFT

API: [`local_cft_analysis`][cylindra.widgets.main.CylindraMainWidget.local_cft_analysis]

### global-CFT

API: [`global_cft_analysis`][cylindra.widgets.main.CylindraMainWidget.global_cft_analysis]

## Lattice Parameters
