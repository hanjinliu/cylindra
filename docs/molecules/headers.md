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
    - [align_all_annealing][cylindra.widgets.sta.SubtomogramAveraging.align_all_annealing]
    - [run_align_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_align_on_landscape]
    - [run_viterbi_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_viterbi_on_landscape]
    - [run_annealing_on_landscape][cylindra.widgets.sta.SubtomogramAveraging.run_annealing_on_landscape]

Following values are measured by [calculate_lattice_structure].

- `"spacing"`
- `"twist"`
- `"skew"`
- `"rise"`
- `"lateral-interval"`
- `"radius"`
- `"curve-index"`
- `"lateral-angle"`
- `"elevation-angle"`
