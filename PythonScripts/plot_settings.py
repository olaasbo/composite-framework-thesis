import matplotlib.pyplot as plt
import numpy as np

def fontSettings(**kwargs):
    import matplotlib as mpl
    import os
    import matplotlib.pylab as pylab
    from matplotlib import font_manager
    # font_manager._rebuild()

    # font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    print("Font settings must be called before the fig is created.")

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 15
    #
    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize

    # define some colors
    teal = (0,0.502,0.502,1)

    font_dirs = (os.path.abspath('') + "\\fonts")  # folder with fonts
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)


    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    if "ppt" in kwargs:
        labcolor = "white"
    else: labcolor = teal
    # print(mpl.rcParams.keys())
    params = {'legend.fontsize':    MEDIUM_SIZE,
              'axes.labelsize':     BIGGER_SIZE,
              'axes.titlesize':     BIGGER_SIZE,
              'axes.labelcolor':    labcolor,
              'axes.titlecolor':    labcolor,
              'xtick.labelsize':    MEDIUM_SIZE,
              'xtick.color':        labcolor,
              'ytick.color':        labcolor,
              'text.color':         labcolor,
              'ytick.labelsize':    MEDIUM_SIZE,
              # 'font.family':           "Tahoma",
                 }

    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # print(mpl.rcParams.keys())
    mpl.rcParams.update(params)

    mpl.rcParams.update({ 'text.color': labcolor})
    # if "ppt" in kwargs:
    #     mpl.rcParams.update({'text.color': labcolor})
        # ticks_font = font_manager.FontProperties(family='Cambria', style='normal',
    return


#latex LaTex LATEX
def LaTeXplot(name="figure", **kwargs):
    """For grouplots, the axis_width is the width of each respective plot.
    """

    import tikzplotlib
    # tikzplotlib.clean_figure()
    savepath = r"PLOTS\\" + name + ".png"
    texpath = r"PLOTS\\" + name + ".tex"
    #
    # tikzplotlib.clean_figure()
    tikzplotlib.save(texpath)
    legend_t = False

    if "plain" not in kwargs:
        print("Adjusting Tex file")
        with open(texpath) as file:
            newlines = []
            clip = False
            continue_reading = False

            # if "axis" in kwargs:
            #     fig, axs = kwargs['axis']
            #     newticks = str([int(i) for i in axs[0].get_yticks()])[1:-1]
            # print("axis" in kwargs)
            for line in file:
                if clip == True:
                    # print("add lines")
                    newlines.append("clip=false, \n")
                    newlines.append("scale only axis, \n")
                    clip = False

                if "center" in kwargs and "begin{tikzpicture}" in line:
                    newline = line.strip() + "[trim axis left, trim axis right]\n"

                elif "begin{axis}" in line or "nextgroup" in line:

                    # if not "clipping" in kwargs:
                    #     print("Clipping made false")
                    #     clip = True
                    newline = line

                elif "tick style" in line and not "tick" in kwargs: # and "axis" in kwargs:
                    newline = (line.split("=")[0] + "={draw=none}," + "\n")
                    newlines.append("scaled " + line[0] + " ticks = false, \n")
                    try:
                        newlines.append("ytick = {"+ newticks +"},\n")
                    except:
                        "newticks does not exist"

                elif "grid style" in line:
                    newline = line.split("}")[0] + "," + " dashed, ultra thin}, \n"

                elif "scale=" in line: # and "clean" in kwargs:
                    # print((line.split("=")[0] + "=1," + "\n"))
                    newline = (line.split("=")[0] + "=1," + "\n")

                elif "mark size=" in line and "msize" in kwargs:
                    segments = line.split("e=")
                    newline = (segments[0] + "e=" + str(kwargs["msize"]) + segments[1][1:] +"")

                # elif "mark size=" in line and "msize" in kwargs:
                #     segments = line.split("e=")
                #     newline = (segments[0] + "e=" + str(kwargs["msize"]) + segments[1][1:] +"")

                elif "legend cell align" in line:
                    newline = (line.split("=")[0] + "={center}," + "\n")
                    legend_t = True

                elif "at" in line and legend_t == True:
                    # if "at" in line:
                    newline = (line.replace("1,", "0.5,"))
                    legend_t = False
                    print(newline)


                elif "clean" in kwargs and continue_reading:
                    newline = ""
                    if "]" in line:
                        continue_reading = False
                        newline=line
                        pass
                    if "min" in line:
                        newline = line


                else:
                    newline = line



                newlines.append(newline)


        with open(texpath, "w") as file:
            for line in newlines:
                file.write(line)
    plt.savefig(savepath)
    return

def ceilm(number,m=10):
    import math
    '''Returns a float rounded up by a factor of the multiple specified'''
    multiple = m ** int(len(str(int(number)))) / 10
    return math.ceil(float(number) / multiple) * multiple

def floorm(number,m=10):
    import math
    '''Returns a float rounded up by a factor of the multiple specified'''
    multiple = m ** int(len(str(int(number)))) / 10
    return math.floor(float(number) / multiple) * multiple




def defaultAxisSettings(ax,arrow=False, font="Cambria", **kwargs):
    """Apply a default style for the layout of the plot axes"""

    fontSettings()
    if "ceilx" in kwargs:
        multiple = kwargs['ceilx']
        print(ax.get_xlim()[1])
        print(multiple, ceilm(ax.get_xlim()[1], multiple))
        ax.set_xlim(left=max(0, int(ax.get_xlim()[0])), right=(ceilm(ax.get_xlim()[1], multiple)))
        # ax.set_xlim(left=max(0, int(ax.get_xlim()[0])))

    if "floorx" in kwargs:
        multiple = kwargs['floorx']

        ax.set_xlim(left=max(0, int(ax.get_xlim()[0])), right=(floorm(ax.get_xlim()[1], multiple)))

    if "ceily" in kwargs:
        multiple = kwargs['ceily']
        topcor = ceilm(ax.get_ylim()[1], multiple)
        print(topcor)
        ax.set_ylim(bottom=min(0, int(ax.get_ylim()[0])), top=(ceilm(ax.get_ylim()[1], multiple)))

    # xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 11)
    # ax.set_xticks(xticks)
    # yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], int(str( ax.get_ylim()[1])[0])+1)
    # yticks = np.delete(yticks, 0)


    # ax.set_yticks(yticks)
    # ax.set_xticklabels(['${}$'.format(int(t)) for t in xticks])
    # ax.set_yticklabels(['${}$'.format(int(t)) for t in yticks])


    # ax.set_xlim(0,100)

    # for labels in [ax.get_yticklabels(), ax.get_xticklabels()]:
    #     for label in labels:
    #         label.set_fontproperties(font) # apply the font to the tick labels

    ax.tick_params(axis=u'both', which=u'both', length=0)  # hide ticks
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_zorder(10)

    if "axes" in kwargs:
        if "both" in kwargs['axes']:
            ax.grid(b=True, which="major", axis="both", linestyle='--', linewidth=1, color="gray", zorder=0)
    else:
        ax.grid(b=True, which="major", axis="x", linestyle='--', linewidth=1, color="gray", zorder=0)



    return


def make_rgb(cols):
    colss = []
    for col in cols:
        new_c = []
        for c in col:
            new_c.append(c / 255)
        new_c = tuple(new_c)
        colss.append(new_c)
    return colss