# Plot gallery
# Some of these plot functions are based on the work of Nils Vedvik:
# Courtesy of Nils. P Vedvik % https://folk.ntnu.no/nilspv/TMM4175/plot-gallery.html
import numpy
import numpy as np


def plotMatIndex(names, xdata, ydata, xlabel, ylabel):
    """Plot certain properties of fibres against eachother"""
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    area = 50
    for k in range(0, len(names)):
        ax.text(xdata[k] * 1.01, ydata[k] * 1.01, names[k])
    ax.grid(True)
    teal = '#008080'
    ax.scatter(xdata, ydata, s=area, c=teal, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title('Title of the plot')
    plt.show(block=False)
    return


# EXAMPLE:
# names=['E-glass', 'T1100', 'M60', 'Steel','Cryptonite']
# xdata=[2.55, 1.79, 1.93, 7.8, 5.0]
# ydata=[76, 324, 588, 201,400]
# # %matplotlib inline
# plotMatIndex(names,xdata,ydata,'Density','Modulus')


def illustrateLayup(layup, number=1, size=(4, 4)):
    """Illustrate the layup of laminate
    Define material, orientation and thickness
    layup = [{mat}:m1, 'ori':0, 'thi':0.5 ]"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    # fig,ax=plt.subplots(figsize=size)
    fig, ax = plt.subplots(figsize=plt.figaspect(0.5), constrained_layout=True)
    tot = 0
    for layer in layup:
        tot = tot + layer['thi']
    hb = -tot / 2.0
    tc = 0
    for layer in layup:
        ht = hb + layer['thi']
        if layer['ori'] == 90:
            # fco = 'lightskyblue'
            #   fco = #7fe7dc
            # fco = (127/255, 231/255,220/255)

            fco = (float(249) / 255, float(229) / 255, float(89) / 255)

            fco = 'gainsboro'
        elif layer['ori'] == 0:
            # fco = 'linen'
            # fco = 'gainsboro'
            # fco = #ced7d8
            # fco = (239/255, 113/255, 38/255)
            # fco = (108/255, 206/255, 203/255)
            fco = (0, float(128) / 255, float(128) / 255)  # teal
        elif layer['ori'] > 0:
            # fco = 'mediumaquamarine'
            fco = (0 / 255, 128 / 255, 128 / 255)
            fco = (float(142) / 255, float(220) / 255, float(157) / 255)  # granny smith apple green
            # fco = (249 / 255, 229 / 255, 89 / 255)
            # fco = 142, 220, 157
        elif layer['ori'] < 0:
            fco = 'lightpink'
            # fco = (244 / 255, 122 / 255, 96 / 255)
        # fc = facecolor, ec = edgecolor
        p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                      clip_on=False, ec='black', fc=fco)
        r = False
        try:
            if "CSM" in layer['mat']['name'] or "CSM" in layer['mat']['abbrev']:
                fco = (float(239) / 255, float(113) / 255, float(38) / 255)  # orange red
                p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                              clip_on=False, ec='black', fc=fco)
        except: pass
        if layer['mat']['abbrev'] == 'C':
            p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, ec='black', fc=fco)
        elif layer['mat']['abbrev'] == 'E':
            p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, ec='black', fc=fco)
        elif layer['mat']['abbrev'] == 'F':
            tc += layer['thi']
            r = True
            # fco = 'papayawhip'
            fco = 'beige'
            p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, linewidth=0, ec='gray', fc=fco, hatch='X')
            q = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, ec='none', fc='None')

        elif layer['mat']['abbrev'] == 'LT':
            r = True
            # fco = 'papayawhip'
            fco = 'yellow'
            p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, linewidth=0, ec='gray', fc=fco, hatch='..-')
            q = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, ec='black', fc='None')
        elif layer['mat']['abbrev'] == 'DB':
            r = True
            # fco = 'papayawhip'
            fco = 'greenyellow'
            p = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, linewidth=0, ec='gray', fc=fco, hatch='..')
            q = Rectangle((-0.6, hb), 1.2, layer['thi'], fill=True,
                          clip_on=False, ec='black', fc='None')

        if r:
            ax.add_patch(p)
            ax.add_patch(q)
        else:
            ax.add_patch(p)
        mid = (ht + hb) / 2.0
        # if layer['ori'] == -45:
        #     pass
        # elif layer['ori'] == 45:
        #     ax.text(0.62, mid, r'$ \pm $' + str(layer['ori']), va='center')
        #     ax.text(-0.68, mid, str(layer['mat']['abbrev']), va='center')  # va = vertical alignment
        h_a, h_t = "right", "left"
        if layer['mat']['abbrev'] == 'F':
            ax.text(-7, mid, str(layer['mat']['abbrev']), va='center', ha=h_t)  # va = vertical alignment
        elif layer['mat']['abbrev'] == 'LT':
            ax.text(-7, mid, "E", va='center', ha=h_t)  # va = vertical alignment
            ax.text(0.68, mid, "0, 90", va='center', ha=h_a)
        elif layer['mat']['abbrev'] == 'DB':
            ax.text(-0.7, mid, "E", va='center', ha=h_t)  # va = vertical alignment
            ax.text(0.68, mid, r'$ \pm 45$', va='center', ha=h_a)
        else:
            ax.text(0.68, mid, str(layer['ori']) + "$\degree$", va='center', ha=h_a)
            ax.text(-0.7, mid, str(layer['mat']['abbrev']), va='center', ha=h_t)  # va = vertical alignment
        hb = ht
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.1 * tot / 2.0, 1.1 * tot / 2.0)
    ax.get_xaxis().set_visible(False)
    ax.plot((-1, -0.8), (0, 0), '--', color='black')
    ax.plot((0.8, 1.0), (0, 0), '--', color='black')

    plottitle = "Laminate"
    try:
        if 'sec' in layup[0]:
            plottitle = (str(layup[0]['sec']) + '\n$t_c$ =' + ("%.3g" % tc) + " mm, $t_f$ = " + ("%.3g" % ((tot - tc)/2))
                         + " mm, $t_{lam}$ = " + ("%.3g" % (tot)) + " mm")
        # + '$t_f$ =' + ("%.3f" % tot-tc) +' mm, ' + str(len(layup)) + ' layers.'
    except:
        plottitle = 'Layup of the laminate - t=' + str(tot) + ' mm, ' + str(len(layup)) + ' layers.'
    # try:
    #     plottitle = 'Layup of the laminate -  (' + layup[0]['rose'] + ')\n' + 't=' + str(tot) + ' mm, ' + str(
    #         len(layup)) + ' layers.'
    # except:
    #     try:
    #         plottitle = 'Layup of the laminate - laminate ' + layup[0]['code'] + '\n' + 't=' + str(tot) + ' mm, ' + str(
    #             len(layup)) + ' layers.'
    #     except:
    #             plottitle = 'Layup of the laminate - laminate ' + str(number+1) + '\n' + 't=' + str(tot) + ' mm, ' + str(
    #                 len(layup)) + ' layers.'
    plt.title(plottitle)
    plt.ylabel('Thickness')
    fig.set_size_inches((8.5, 11), forward=False)
    # plt.figure(figsize=plt.figaspect(0.5))
    plt.show(block=False)


def plotLayerStresses(results, loadcase, plotstrains=False, **kwargs):
    from plot_settings import defaultAxisSettings
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    layup = results
    mm = (1 / 2.54)  # cm in inches

    z, ori = [], []
    sx, sy, sxy = [], [], []
    s1, s2, s12 = [], [], []
    ex, ey, exy = [], [], []
    # print(results[0]['strain'])
    bot = -sum([layer["thi"] for layer in layup]) / 2

    for i, layer in enumerate(layup):
        top = bot + layer["thi"]
        z.append(bot)
        z.append(top)

        ex.append(layer['strain']['xyz']['bot'][0])
        ex.append(layer['strain']['xyz']['top'][0])
        ey.append(layer['strain']['xyz']['bot'][1])
        ey.append(layer['strain']['xyz']['top'][1])
        exy.append(layer['strain']['xyz']['bot'][2])
        exy.append(layer['strain']['xyz']['top'][2])

        sx.append(layer['stress']['xyz']['bot'][0])
        sx.append(layer['stress']['xyz']['top'][0])
        sy.append(layer['stress']['xyz']['bot'][1])
        sy.append(layer['stress']['xyz']['top'][1])
        sxy.append(layer['stress']['xyz']['bot'][2])
        sxy.append(layer['stress']['xyz']['top'][2])
        s1.append(layer['stress']['123']['bot'][0])
        s1.append(layer['stress']['123']['top'][0])
        s2.append(layer['stress']['123']['bot'][1])
        s2.append(layer['stress']['123']['top'][1])
        s12.append(layer['stress']['123']['bot'][2])
        s12.append(layer['stress']['123']['top'][2])

        bot = top

    # print(mm)
    # plt.rcParams["figure.figsize"] = 14.9 * mm, 10.0 * mm
    # plt.rcParams["figure.figsize"] = 5,10
    n = 2
    if plotstrains:
        n = 3
    n = 4
    # fig,axs = plt.subplots(ncols=3,nrows=1)

    fig = plt.figure()

    gs = fig.add_gridspec(1, 6)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    axs = [ax1, ax2, ax3]
    tl = 2

    colors = ["red", "blue", "green"]
    fillalpha = 0.1
    #
    #
    for n, ax in enumerate(axs):

        if plotstrains:
            if n == 0:
                for i, (res, lab) in enumerate(
                        zip([ex, ey, exy], ['$\\varepsilon_{x}$', '$\\varepsilon_{y}$', '$\\gamma_{xy}$'])):
                    if "axial" in kwargs:
                        if i == 1: break

                    ax.plot(res, z, '-', lw=tl, color=colors[i], zorder=10, label=lab)
                    ax.fill_betweenx(z, res, x2=0, alpha=fillalpha, fc=colors[i], zorder=9)
                for tick in ax.get_xticklabels():
                    # tick.set_rotation(45)
                    ax.set_xlabel('Strains in structural coordinates', zorder=2)
                    ax.legend(loc="lower right").set_zorder(20)
                    # ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 1e3 ))
                    # ax.xaxis.set_major_formatter(ticks)
            if n == 1:
                for i, (res, lab) in enumerate(zip([sx, sy, sxy], ['$\sigma_{x}$', '$\sigma_{y}$', '$\\tau_{xy}$'])):
                    if "axial" in kwargs:
                        if i == 1: break

                    ax.plot(res, z, '-', lw=tl, color=colors[i], zorder=10, label=lab)
                    ax.fill_betweenx(z, res, x2=0, alpha=fillalpha, fc=colors[i], zorder=9)
                    ax.set_xlabel('Global stresses \n($N/$mm$^2$)', zorder=2)
                    ax.legend(loc="lower right").set_zorder(20)

            if n == 2:
                for i, (res, lab) in enumerate(zip([s1, s2, s12], ['$\sigma_{1}$', '$\sigma_{2}$', '$\\tau_{12}$'])):
                    if "axial" in kwargs:
                        if i == 1: break

                    ax.plot(res, z, '-', lw=tl, color=colors[i], zorder=10, label=lab)
                    ax.fill_betweenx(z, res, x2=0, alpha=fillalpha, fc=colors[i], zorder=9)

                    ax.set_xlabel('Local stresses \n($N/$mm$^2$)', zorder=2)
                    ax.legend(loc="lower right").set_zorder(20)
                    # if i != 0:
                    #     ax.scatter(res, z, lw=tl, color=colors[i], zorder=10)
                    #     for x,y in zip(res,z):
                    #         ff = "%.4f" % x
                    #         ax.annotate(ff, (x,y), (-2,-2),xycoords="data", textcoords="offset points", color=colors[i], clip_on=False,
                    #                     backgroundcolor="white", zorder=1000,rotation=45)

    order = 15
    yticks = (sorted(list(set(z))))

    title = "Loads: "
    # if loadcase:
    #     lc = loadcase[0]
    #     for key in lc:
    #         title += key + "=" + str(lc[key]) + " "

    axs[0].set_ylabel("z (mm)")

    for i, ax in enumerate(axs):
        defaultAxisSettings(ax)
        if i == 0:
            ax.set_yticks(yticks)
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(yticks[0], yticks[-1])

        for i, loc in enumerate(yticks):
            if loc != 0 and loc != max(yticks) and i!=0:
                ax.axhline(y=loc, linestyle='--', linewidth=1, color="gray", zorder=0,)

        # ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=len(axs))
    #

    ypos = (axs[0].get_yticks())
    xmin = (axs[0].get_xlim()[0])
    #
    # # in case the stresses/strains are all positive or negative,
    # # we dont want to make this shift the plot out of the focus area.
    # # therefore, we draw the vertical bar indicating 0 last
    for i, ax in enumerate(axs):
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax)
        ax.grid(axis="x",zorder=0,linestyle="--", linewidth=1)
        ax.plot([0, 0], [yticks[0], yticks[-1]], c="k", zorder=5)
        ax.plot([xmin, xmax], [0, 0], "k", linestyle="dashdot", zorder=5)
        # if i == 0:
        #     # ax.text((xmax - abs(xmin)) / 2, 0, 'N.A.', color="k", backgroundcolor="white",
        #     #         horizontalalignment='center',
        #     #         verticalalignment='center',
        #     #         zorder=6)
        #     ax.annotate('Centre line', xy=(1,0.5), xycoords='axes fraction',
        #                 xytext=(1, 1), textcoords='offset points',
        #                 backgroundcolor="w", zorder= 1000, rotation = 45,
        #                 arrowprops=dict(arrowstyle="-"),
        #                 horizontalalignment='left', verticalalignment='bottom', clip_on=False
        #                 )


    for i, (y, layer) in enumerate(zip(ypos, layup)):
        if i == len(ypos) - 1: break
        textpos = ypos[i] + abs((ypos[i + 1] - ypos[i]) / 2)
        tag = str(layer['ori']) + '$\degree$'
        xmin, xmax = (axs[-1].get_xlim())
        axs[-1].annotate(tag, xy=(xmax, textpos), xycoords="data",
                         xytext=(xmax * 1.30, textpos), ha="right", va="center", zorder=20)
        # axs[-1].text(xmax+5,textpos, tag, ha="right", va="center", backgroundcolor="white",zorder=21,clip_on=False)
    # for ax in axs:
    #     for key, spine in ax.spines.items():
    #         spine.set_visible(False)
    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()
    textdic = {"va": "center",'xycoords':"data", 'textcoords': "data", "ha": "center", "zorder": 14}
    label = "\scriptsize{centre line}"
    ax2.annotate(label, (xmax*0.8,0), (xmax, 1.4), rotation=45, color="black",
                 # arrowprops=dict(arrowstyle="-", color="black"),
                 bbox=dict(boxstyle="larrow,pad=0.1", fc="white", ec="black", lw=1),
                 # arrowprops=dict(arrowstyle="->",
                 #                 connectionstyle="angle, angleA=1,angleB=80"),
                 **textdic)



    fig.subplots_adjust(hspace=0, wspace=0.1)
    plt.show(block=False)
    return fig, axs


def plotLayerFailure(layup):
    from plot_settings import defaultAxisSettings
    import matplotlib.pyplot as plt
    z = []
    ms, me, tw = [], [], []
    bot = -sum([layer["thi"] for layer in layup]) / 2

    for i, layer in enumerate(layup):
        top = bot + layer["thi"]
        z.append(bot)
        z.append(top)

        # maximum stress
        ms.append(layer['fail']['MS']['bot'])
        ms.append(layer['fail']['MS']['top'])

        me.append(layer['fail']['ME']['bot'])
        me.append(layer['fail']['ME']['top'])

        tw.append(layer['fail']['TW']['bot'])
        tw.append(layer['fail']['TW']['top'])
        ratio = (max([layer['fail']['TW']['top']/layer['fail']['MS']['top'], layer['fail']['TW']['top']/layer['fail']['ME']['top']]))

        print("%-10.i %3s TW: %.3f MS: %.3f ME: %.3f %.3f  " % (i, layer['ori'],
                                                                 layer['fail']['TW']['top'],
                                                                 layer['fail']['MS']['top'],
                                                                 layer['fail']['ME']['top'], ((ratio - 1) * 100)))

        bot = top

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 4))
    yticks = (sorted(list(set(z))))
    tl = 2
    ax.plot(ms, z, '-', color='blue', label='$f_E (MS)$', lw=tl, zorder=10)
    ax.plot(me, z, '-', color='green', label='$f_E (ME)$', lw=tl, zorder=10)
    ax.plot(tw, z, '-', color='red', label='$f_E (TW)$', lw=tl, zorder=10)
    ax.set_xlabel("EXPOSURE FACTOR, $f_E$")
    ax.set_ylabel("$z$ (mm)")
    ax.legend(loc="lower right").set_zorder(8)

    ax.set_yticks(yticks)

    defaultAxisSettings(ax, ceil=1)
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(0, 0.7)
    ax.set_xticks(np.arange(0, 0.7, 0.1))
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], "k", linestyle="dashdot", zorder=5)
    print("Xlim", ax.get_xlim()[1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.grid(axis="x", zorder=0, linestyle="--", linewidth=1)
    for i, loc in enumerate(yticks):
        if loc != 0 and loc != max(yticks) and i != 0:
            ax.axhline(y=loc, linestyle='--', linewidth=1, color="gray", zorder=0, )
    ypos = (ax.get_yticks())
    xmin, xmax = (ax.get_xlim())
    for i, (y, layer) in enumerate(zip(ypos, layup)):
        if i == len(ypos) - 1: break
        textpos = ypos[i] + abs((ypos[i + 1] - ypos[i]) / 2)
        # print(ypos[i], ypos[i + 1], (ypos[i] + textpos))
        tag = str(layer['ori']) + '$\degree$'
        # for ax in axs[-2:-1]:
        xmax = (ax.get_xlim()[1])
        ax.annotate(tag, xy=(xmax, textpos), xycoords="data",
                    xytext=(xmax * 1.08, textpos), ha="right", va="center")



    plt.tight_layout()
    plt.show(block=False)
    return fig, [ax]


# Failure envelopes
linewidth = 3


def plotTsaiWu(ax, material, lw=3):
    # from plot_settings import defaultAxisSettings
    import matplotlib.pyplot as plt
    import numpy as np
    from math import cos, sin, radians
    from failure_criterion import TsaiWu
    s1, s2 = [], []

    for a in np.linspace(0, 360, 1000):
        s1i = cos(radians(a))
        s2i = sin(radians(a))
        fE = TsaiWu((s1i, s2i, 0, 0, 0, 0), material)
        s1.append(s1i / fE)
        s2.append(s2i / fE)
    ax.plot(s1, s2, 'r-', label="Tsai-Wu", zorder=20, lw=lw)
    return s1, s2, ax


def plotMaxStrain(ax, material, col="blue", lw=3):
    from failure_criterion import maximumStrain
    from plot_settings import defaultAxisSettings
    import matplotlib.pyplot as plt
    import numpy as np
    from math import cos, sin, radians
    s1, s2 = [], []

    for a in np.linspace(0, 360, 3600):
        s1i = cos(radians(a))
        s2i = sin(radians(a))
        fE = maximumStrain((s1i, s2i, 0, 0, 0, 0), material)
        s1.append(s1i / fE)
        s2.append(s2i / fE)
    ax.plot(s1, s2, '--', c=col, label="Maximum Strain", zorder=20, lw=lw)
    return s1, s2


def plotMaxStress(ax, material, col="blue", lw=3):
    from failure_criterion import maximumStress
    from plot_settings import defaultAxisSettings
    # import matplotlib.pyplot as plt
    import numpy as np
    from math import cos, sin, radians
    s1, s2 = [], []
    for a in np.linspace(0, 360, 2000):
        s1i = cos(radians(a))
        s2i = sin(radians(a))
        fE = maximumStress((s1i, s2i, 0, 0, 0, 0), material)
        s1.append(s1i / fE)
        s2.append(s2i / fE)
    ax.plot(s1, s2, '--', c=col, label="Maximum Stress", zorder=20, lw=lw)
    return s1, s2


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plotFailureEnvelopes(material, **kwargs):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from math import cos, sin, radians
    from plot_settings import defaultAxisSettings
    fig, ax = plt.subplots(1)
    try:
        cut = kwargs['cut']
    except: cut = False

    s1, s2 = [], []

    if cut:
        ax.set_ylim(bottom=-10, top=60)
        ax.set_xlim(left=-80, right=10)
    if "maxstress" in kwargs["plot"]:
        col = "blue"
        s1, s2 = plotMaxStress(ax, material, col)

        ss = list(zip(s1, s2))
        anndict = {"color": col, "backgroundcolor": "w",
                   "horizontalalignment": 'center', "verticalalignment": 'bottom', "clip_on": "False",
                   "zorder": 100}
        if not cut:
            sx = [ss[i] for i,x in enumerate(ss) if ss[i][0] <= 0 and ss[i][1] > 0]

            s11 = [sx[i][0] for i in range(len(sx))]
            s22 = [sx[i][1] for i in range(len(sx))]

        else:
            sx = [ss[i] for i,x in enumerate(ss) if ss[i][0] <= 0 and ss[i][1] > 0]

            s11 = [sx[i][0] for i in range(len(sx))]
            s22 = [sx[i][1] for i in range(len(sx))]

        if not cut:
            ax.annotate('Max Stress criterion', xy=(min(s1) - 1, max(s2) + 1), xycoords='data',
                        xytext=(min(s1) - 20, max(s2) + 10), textcoords='data',
                        color=col, backgroundcolor="w",
                        arrowprops=dict(arrowstyle="-", color=col),
                        horizontalalignment='center', verticalalignment='bottom', clip_on=False
                        )
        else:
            s11c = find_nearest(s11, ax.get_xlim()[0])
            index = s11.index(s11c)
            ax.annotate('Max Sress criterion', xy=(s11[index-2], s22[index-2]), xycoords='data',
                        arrowprops=dict(arrowstyle="-", color=col),
                        xytext=(s11[index], s22[index] + 7), textcoords='data', va="bottom",
                        **anndict)

    if "maxstrain" in kwargs["plot"]:
        col = "green"
        s1, s2 = plotMaxStrain(ax, material, col)
        ss = list(zip(s1, s2))

        if not cut:
            sx = [ss[i] for i,x in enumerate(ss) if ss[i][0] >= 0 and ss[i][1] > 0]

            s11 = [sx[i][0] for i in range(len(sx))]
            s22 = [sx[i][1] for i in range(len(sx))]


        else:
            sx = [ss[i] for i,x in enumerate(ss) if ss[i][0] <= 0 and ss[i][1] > 0]

            s11 = [sx[i][0] for i in range(len(sx))]
            s22 = [sx[i][1] for i in range(len(sx))]

        xmin, xmax = min(s11), max(s11)
        s1_copy = (xmin + xmax) / 1.5
        s11c = find_nearest(s11, s1_copy)
        index = s11.index(s11c)
        anndict = {"color": col, "backgroundcolor": "w",
                   "horizontalalignment": 'center', "verticalalignment": 'bottom', "clip_on": "False",
                   "zorder": 100}

        if not cut:
            s2_copy = max(s22)
            s22cc = find_nearest(s11, max(s22))
            index = s22.index(max(s22))
            ax.annotate('Max Strain criterion', xy=(s11[index], s22[index]), xycoords='data',
                        xytext=(s11[index]+10, s22[index]+10), textcoords='data',
                        arrowprops=dict(arrowstyle="-", color="green"),
                                                **anndict)
        else:
            s11c = find_nearest(s11, ax.get_xlim()[0])
            index = s11.index(s11c)
            ax.annotate('Max Strain criterion', xy=(s11[index-2], s22[index-2]), xycoords='data',
                        arrowprops=dict(arrowstyle="-", color="green"),
                        xytext=(s11[index] , s22[index]-10 ), textcoords='data', va="top",
                        **anndict)
    if "tsaiwu" in kwargs["plot"]:
        col ="red"
        s1, s2, ax = plotTsaiWu(ax, material)
        ss = list(zip(s1, s2))

        if not cut:
            sx = [ss[i] for i,x in enumerate(ss) if ss[i][0] >= 0 and ss[i][1] < 0]

            s11 = [sx[i][0] for i in range(len(sx))]
            s22 = [sx[i][1] for i in range(len(sx))]

        else:
            sx = [ss[i] for i,x in enumerate(ss) if ss[i][0] <= 0 and ss[i][1] > 0]

            s11 = [sx[i][0] for i in range(len(sx))]
            s22 = [sx[i][1] for i in range(len(sx))]

        xmin, xmax = min(s11), max(s11)
        s1_copy = (xmin + xmax) / 2.3
        s11c = find_nearest(s11, s1_copy)
        index = s11.index(s11c)
        anndict = {"color":col, "backgroundcolor":"w",
                        "horizontalalignment":'center', "verticalalignment":'top', "clip_on":"False",
                        "zorder":100}
        if not cut:
            ax.annotate('Tsai-Wu criterion', xy=(s11[index], s22[index]), xycoords='data',
                        arrowprops=dict(arrowstyle="-", color=col),
                        xytext=(s11[index] + 10, s22[index] - 70), textcoords='data',
                        **anndict)

        else:
            xmin, xmax = ax.get_xlim()
            s1_copy = (xmin + xmax) / 2.3
            s11c = find_nearest(s11, s1_copy)
            index = s11.index(s11c)
            ax.annotate('Tsai-Wu criterion', xy=(s11[index], s22[index]), xycoords='data',
                        arrowprops=dict(arrowstyle="-", color=col),
                        xytext=(s11[index]-2, s22[index]+5), textcoords='data', va="bottom",
                        **anndict)





    xmax, ymax = ax.get_xlim()[-1], ax.get_ylim()[-1]
    ax.annotate("$Y_T$", xy=(0 + 2, material["YT"] + 2), color="k", va="bottom", ha="left", zorder=100)
    ax.text(0, ymax + 5, "$\sigma_2$", va="bottom", ha="center", color='k')

    ax.text(xmax + 1, 0, "$\sigma_1$", va="center", ha="left", color='k')

    if not cut:
        ax.annotate("$Y_C$", xy=(0 + 1, -material["YC"] - 10), color="k", va="top", ha="left", zorder=100)
        ax.annotate("$X_T$", xy=(material["XT"] + 10, +2), color="k", va="bottom", ha="left", zorder=100)
        ax.annotate("$X_C$", xy=(-material["XC"] - 10, +2), color="k", va="bottom", ha="right", zorder=100)

    for loc in ["right", "top"]:
        ax.spines[loc].set_visible(False)
    defaultAxisSettings(ax)
    ax.grid(False)
    ax.spines['left'].set_position(("data", 0.0))
    ax.spines['bottom'].set_position(("data", 0.0))
    return ax
