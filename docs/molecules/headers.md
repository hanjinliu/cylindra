# Feature Names

Here's the list of the standard feature names and how they will be added.

- `"nth"`

    The n-th molecule of a protofilament starting from 0. If this value is 3, it means the
    molecule is the 4th molecule from the tip (= the starting edge of the source spline).

    This feature is added by:

    - [map_monomers][cylindra.widgets.main.CylindraMainWidget.map_monomers]
    - [map_monomers_with_extensions][cylindra.widgets.main.CylindraMainWidget.map_monomers_with_extensions]

- `"pf-id"`

    The protofilament ID. Molecules labeled with the same `pf-id` belong to the same
    protofilament.

    This feature is added by:

    - [map_monomers][cylindra.widgets.main.CylindraMainWidget.map_monomers]
    - [map_monomers_with_extensions][cylindra.widgets.main.CylindraMainWidget.map_monomers_with_extensions]

- `"position"`

    The position of the molecule in the spline coordinate (in nm). This value is useful
    when you want to know in which position the molecule is located along the spline.

    This feature is added by:

    - [map_monomers][cylindra.widgets.main.CylindraMainWidget.map_monomers]
    - [map_monomers_with_extensions][cylindra.widgets.main.CylindraMainWidget.map_monomers_with_extensions]

- `"isotype-id"`

    The isotype ID. Molecules labeled with the same `isotype-id` belong to the same
    isotype.

    This feature is added by:

    - [seam_search][cylindra.widgets.sta.SubtomogramAveraging.seam_search]
    - [seam_search_by_feature][cylindra.widgets.sta.SubtomogramAveraging.seam_search_by_feature]
    - [seam_search_manually][cylindra.widgets.sta.SubtomogramAveraging.seam_search_manually]

- `"score"`

    The alignment score. Higher scores indicate better alignment. Note that the score
    may be affected by the missing wedge.

    This feature is added by:

    - [align_all][cylindra.widgets.sta.SubtomogramAveraging.align_all]
    - [align_all_template_free][cylindra.widgets.sta.SubtomogramAveraging.align_all_template_free]
    - [align_all_viterbi][cylindra.widgets.sta.SubtomogramAveraging.align_all_viterbi]
    - [align_all_rma][cylindra.widgets.sta.SubtomogramAveraging.align_all_rma]
    - [align_all_rfa][cylindra.widgets.sta.SubtomogramAveraging.align_all_rfa]
    - [run_align_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_align_on_landscape]
    - [run_viterbi_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_viterbi_on_landscape]
    - [run_rma_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_rma_on_landscape]
    - [run_rfa_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_rfa_on_landscape]

Following values are measured by [calculate_lattice_structure][cylindra.widgets.main.CylindraMainWidget.calculate_lattice_structure].

- `"spacing"`: projected distance between the longitudinally adjacent molecules.
- `"twist"`: the twist angle between the longitudinally adjacent molecules in degrees.
- `"skew"`: the projected skew angle between the longitudinally adjacent molecules in degrees.
- `"rise"`: the projected rise angle between the laterally adjacent molecules in degrees.
- `"lateral-interval"`: the projected distance between the laterally adjacent molecules.
- `"radius"`: the radius as a cylinder measured from the spline.
- `"curve-index"`: a quantity &eta; defined between -1 and 1 that represents whether the
  molecule is inside or outside the curve. &eta; is calculated by taking the dot product
  of the second derivative of the spline and the vector from the spline to the molecule.
  Therefore, &eta; > 0 means the molecule is inside the curve, and &eta; < 0 means the
  molecule is outside the curve.
- `"lateral-angle"`: the angle formed by the left and right laterally adjacent molecules.
- `"elevation-angle"`: the angle formed by the vector pointing to the longitudinally
  adjacent molecule and the spline tangent vector.
