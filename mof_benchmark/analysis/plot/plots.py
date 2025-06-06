from tueplots import axes, cycler, figsizes, fonts, fontsizes
from tueplots.constants.color import palettes, rgb

def style(*, column="half", nrows=1, ncols=1, usetex=True, family="serif", half_factor = 0.75, full_factor=0.6):
    """ICML 2024 bundle."""
    if column == "half":
        size = figsizes.icml2024_half(nrows=nrows, ncols=ncols)
        size['figure.figsize'] = tuple([el * half_factor for el in size['figure.figsize']])
    elif column == "full":
        size = figsizes.icml2024_full(nrows=nrows, ncols=ncols)
        size['figure.figsize'] = tuple([el * full_factor for el in size['figure.figsize']])
    else:
        msg = _msg_error_wrong_arg_column(column)
        raise ValueError(msg)

    
    if usetex is True:
        font_config = fonts.icml2024_tex(family=family)
    elif usetex is False:
        font_config = fonts.icml2024(family=family)
    else:
        raise ValueError(_msg_error_wrong_arg_usetex(usetex))
    fontsize_config = fontsizes.icml2024()
    fontsize_config["text.latex.preamble"] = r"\usepackage{times} \renewcommand{\familydefault}{\sfdefault} \usepackage{sansmath} \sansmath \usepackage{upgreek}"
    
    return {
        **font_config,
        **size,
        **fontsize_config,
        **cycler.cycler(color=palettes.tue_plot),
        **axes.lines(),
        'figure.dpi': 144,
    }

