# Subtomogram Alignment and Averaging

The subtomogram alignment and the subsequent averaging is the important part of the
cryo-ET analysis workflows. `cylindra` provides methods to perform the analyses. In the
GUI, these methods are all in a separate dock widget titled "STA widget". You can open
via `Analysis > Open STA widget` (++ctrl+k++ &rarr; ++s++).

![STA widget](../images/sta_widget.png){ loading=lazy, width=480px }

Since many methods share the same parameters, the STA widget uses the same widget for
these parameters.

- **Template**: The template image. This parameter can be a path to an image file, a
  list of paths for multi-template alignment, or use the last average image. Template
  images don't have to be the same scale as the subtomogram images. They will be
  rescaled to the same scale as the (binned) tomogram image.
- **Mask**: The mask image. You can create a mask by blurring the template, or supply
  a mask image. To blur the template, you need to specify the "dilate radius" and
  "sigma" parameters. These parameters are used to dilate the binarized template image
  and soften the edges by Gaussian blurring.

## Index

- [Conventional Methods in Cryo-ET Studies](conventional.md)
- [Build Correlation Landscapes](landscape.md)
- [Viterbi Alignment](viterbi.md)
- [Restricted Mesh Annealing (RMA)](rma.md)
