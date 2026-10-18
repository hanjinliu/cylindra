# Checking Tomogram Handedness

Tomogram handedness can be quickly checked by the lattice periodicity. If the tomogram
is reconstructed using the wrong handedness, cylindrical power spectrum will be flipped.
If the expected peak is higher in the flipped spectrum, the tomogram is likely
reconstructed with the wrong handedness.

1. Open a tomogram or a tilt series (see [Open an Image](../open_image.md#open-an-image)).
2. Select the appropriate spline configuration for the lattice you want to check. Default is eukaryotic microtubule.
3. Draw a spline and roughly fit it. [Manual fitting](../fit_splines.md#fit-manually) is usually sufficient.
4. Run `Analysis > Utilities > Check handedness`. The result will be printed in the log.
