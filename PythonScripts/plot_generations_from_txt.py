import pandas as pd
import os
import plotly.express as px
import matplotlib as mpl
from matplotlib import pyplot as plt
import copy
import numpy as np
from plot_settings import fontSettings, defaultAxisSettings


pd.set_option("display.max_rows", None, "display.max_columns", None)
from matplotlib.ticker import FuncFormatter

"""The output from the Galapagos record sorts the genomes on their function value, i.e. from 
low to high."""
txt_file = r"C:\Users\olaas\Desktop\GirderOptimisation\GalapagosRecord.txt"
txt_file = r"C:\Users\olaas\Desktop\Klosterbrua_Sensitivity\Galapagos22.txt"

# C:\Users\olaas\Desktop\Klosterbrua_lp_opt\Galapogos10.txt


path = r"C:\Users\olaas\Desktop\Klosterbrua_lp_opt\sensitivity_20.xls"

path = r"C:\Users\olaas\Desktop\Klosterbrua_Sensitivity\sensitivityc2.xls"
path = r"C:\Users\olaas\Desktop\Klosterbrua_Sensitivity\Globalw10w21c.xls"

path = r"C:\Users\olaas\Desktop\OptimOption\optimopt1.xls"
results = pd.read_excel(path)


# NN = results.dropna()
# fig10,ax10 = plt.subplots(1)
#
# n = range(len(NN))

# ax10.scatter(n, NN["cost"],s=10)
# ax10.set_ylim(bottom=0.5, top=1.0)
def mean_value(list):
    if len(list) == 0:
        return None
    else:
        return sum(list) / len(list)


def removeDuplicates(df, pos="last", **kwargs):
    """Removes consecutive entries in the selected columns.
    The corresponding row is removed.
    :param df: a (pandas) dataframe
    :param cols: the columns to check, cols = ["a","b", ...]
    :param pos: keep the "first" or the "last" row

    :return a cleaner dataframe
    """

    if 'names' in kwargs:
        cols = kwargs['names']

        print(cols)
        if pos == "last":
            return df.loc[(df[cols].shift(-1) != df[cols]).any(axis=1)]
        # else:
        #     return df.loc[(df[cols].shift(-1) != df[cols]).any(axis=1)]


with open(txt_file) as file:
    d = {}  # create dataframe for pandas
    generations = []
    genomes = 0
    fitnesses = []
    for line in file:
        if "Generation" in line:
            generation = int(line.split(" ")[1].strip())
            try:
                d['generations'].append(generation)
            except:
                d['generations'] = [generation]
            if fitnesses:
                best_fitness = min(fitnesses)
                mean_fitness = mean_value(fitnesses)
                # print("%-10i %10.2f %10.2f %10.2f" % (generation, mean_fitness, best_fitness ,max(fitnesses)))

                try:
                    d['best'].append(best_fitness)
                    d['mean'].append(mean_fitness)
                    d['genomes'].append(genomes)
                except:
                    d['best'] = [best_fitness]
                    d['mean'] = [mean_fitness]
                    d['genomes'] = [genomes]
                genomes = 0
                fitnesses = []
            continue
        if "Fitness" in line:
            genomes += 1
            fitness = line.split("Fitness=")[1].split(",")[0]
            if fitness != "NaN":
                fitnesses.append(float(fitness))

# the values corresponding to the last generation must be added
best_fitness, mean_fitness = min(fitnesses), mean_value(fitnesses)
d['best'].append(best_fitness)
d['mean'].append(mean_fitness)
d['genomes'].append(genomes)

# plot from text file

lowest_fitness = [d['best'][0]]
for i, fit in enumerate(d['best'][1::]):
    fit, prev = float(fit), lowest_fitness[i - 1]
    if fit < prev:
        lowest_fitness.append(fit)
    else:
        lowest_fitness.append(prev)

d['best'] = lowest_fitness

# transform dictionary to dataframe, then plot using pandas
df = pd.DataFrame(data=d)

columns_to_read = ['fitness', 'cost', 'footprint']

rows_start, rows_end = 0, 0

# new_dataframe = {'generation': []}
# for col in columns_to_read:
#     for n in ["max_", "min_", "std_", "mean_"]:
#         entry = (n + col).strip()
#         new_dataframe[entry] = []  # add empty lists to the dictionary

# ------------------------------------------------------------------------
# ppt=True

if "ppt" in globals():
    fontSettings(ppt=True)
else: fontSettings()
fig1,(ax11,ax22, ax33, ax44) = plt.subplots(4, sharex=True, figsize=(1260*0.0104166667, 720*0.0104166667))

fig2,(ax1) = plt.subplots(1, figsize=(1260*0.0104166667, 720*0.0104166667))
# fig3, ax22 = plt.subplots(1, figsize=(1260*0.0104166667, 720*0.0104166667))
fig3, ax100 = plt.subplots(1, figsize=(1260*0.0104166667, 720*0.0104166667))

worst = [i for i,x in enumerate(d['best']) if x >= 1]
worst.append(worst[-1]+1)
j = max(worst)
least_fit, first_gen = [d['best'][i] for i in worst], [d['generations'][i] for i in worst]
best_fit, last_gen = d['best'][j:len(d['generations'])], d['generations'][j:len(d['generations'])]
ax1.scatter(d['generations'][j],d['best'][j],fc="none", ec="k",zorder=50)


fittest = [x for x in d['best'] if x <=1]

ant = "Gen:" + str(d['generations'][-1]) + "; Fitness:" + str(d['best'][-1])
ax1.annotate(ant, (d['generations'][-1],d['best'][-1]),(0,30), textcoords="offset points",
             backgroundcolor="white", ha="right", va="center",arrowprops=dict(arrowstyle="->"),zorder=100,fontsize=12)

# ant = "Gen:" + str(d['generations'][j]) + "; Fitness:" + str(d['best'][j])
# ax1.annotate(ant,(d['generations'][j],d['best'][j]),(30,30), textcoords="offset points", ha="left", va="center",
#              backgroundcolor="white", arrowprops=dict(arrowstyle="->"),zorder=100,fontsize=12)

# ax1.scatter(first_gen,least_fit, fc="r", ec="k")
ax1.plot(first_gen,least_fit, "r--",zorder=10)
# ax1.scatter(last_gen, best_fit,fc="r", ec="k")
ax1.plot(last_gen, best_fit,"r",zorder=10)
ax1.scatter(last_gen[-1], best_fit[-1],marker="x",fc="k",ec="k",zorder=50)

ax1.set_title("FITNESS vs. GENERATIONS")
ax1.set_xlabel("GENERATIONS")
# ax11.set_ylabel("FITNESS (penalty fitness value)")

# defaultAxisSettings(ax11,axes="both")

# ax11.set_ylim(0.8,0.9)
# ax11.set_xlim(0,250)

# -----------------------------------------------------------------------

df = results.dropna()
df = removeDuplicates(df, names=df.columns.tolist()[1::])
# df = removeDuplicates(df,names=["fitness","cost"])

# define run time in hours
df.time = (df.time - df.time.tolist()[0]) / (60*60)
df.rename(columns = {'time':'hours'}, inplace = True)


gens, fits = [], []
current_best = df['fitness'].tolist()[1]


for i in range(len(df)):
    if i <= 160:
        g = int((i-1)/160)+1
    elif i >160:
        g = ((int((i - 1) / 80) + 1)-1)
    gens.append(g)


current_best_copy, feasible = copy.copy(current_best), []
for fval, wval in zip(df['fitness'].tolist(), df['wsum'].tolist()):
    feas = "Null"
    if current_best > fval:
        current_best = fval
        if fval == wval:
            feas = fval


    fits.append(current_best)
    feasible.append(feas)


df.insert(0,"Generation",gens)
df.insert(1,"BestFit", fits)
df.insert(2,"Feasible", feasible)

if "ppt" in globals():
    linecolor="white"
else: linecolor="gray"
tinydf = df[["hours","Generation","BestFit","Feasible","wsum","cost","footprint"]]  # simplified dataframe
# tinydf = removeDuplicates(tinydf, pos="last", names=["Generation"]) # keep only the best row in each generation

feasibledf = tinydf.drop(tinydf[tinydf.Feasible=="Null"].index)
tinydf = removeDuplicates(tinydf, pos="last", names=["Generation"]) # keep only the best row in each generation
# print(tinydf[["hours", "Generation","BestFit","Feasible","wsum"]])
print(feasibledf)

tinydf.plot(x="Generation",  y="BestFit", ax=ax11, color=linecolor, linewidth=2, label="Fitness")
tinydf.plot(x="Generation",  y=["footprint","cost"], ax=ax33, color=["blue","red"], linewidth=2, label=["Footprint","Cost"])

ax44.plot(tinydf.Generation,(tinydf["cost"]/tinydf["footprint"]), color="green")



# tinydf.plot(x="hours",  y="BestFit", ax=ax33, color="none", linewidth=0)
# tinydf.plot(x="Generation", y="Bestfit", ax=ax33, color=linecolor, linewidth=1, label="Cost")
fig,ax111 = plt.subplots(1)
ax111.set_title("Fitness value when all the constraints are satisfied")
ax111.scatter(feasibledf.Generation, feasibledf.Feasible,s=20,ec="black",zorder=10, label="Feasible")
ax111.plot(tinydf.Generation,tinydf.BestFit,zorder=9,c=(0,125/255,125/255,0.5),label="Infeasible")
ax111.legend(loc="best")

ax11.set_title("Fitness value")
ax22.set_title("Fitness value when all the constraints are satisfied")
ax33.set_title("Cost - Carbon footprint")
ax33.set_ylabel("[kgCO$_2$e], [€]",color="blue")
ax44.set_title("Ratio between objectives: $\\frac{\mathrm{Footrprint}}{Cost}$")
# the fitness when all constraints are satisfied correspond to the values when fitness = weighted sum
ax22.plot(feasibledf.Generation, feasibledf.BestFit, color=linecolor, lw=2)
# ax101.plot(feasibledf.Generation, feasibledf.BestFit, color=linecolor, lw=2)
ax100.plot(feasibledf.hours,feasibledf.BestFit, color=linecolor, lw=2)
ax101=ax100.twiny()
ax100.set_xlabel("Hours")
ax101.set_xlabel("Generations")
# ax1.scatter(last_gen[-1], best_fit[-1],marker="x",fc="k",ec="k",zorder=50)
# tinydf.plot(x="hours", y="BestFit", ax=ax11, color="green", linewidth=2)
# tinydf.plot.scatter(x="hours", y="BestFit", ax=ax11, color="green")
# tinydf.plot(x="hours", y="BestFit", ax=ax22)
minbf = current_best_copy
non_improvement_counter = 0
label_added1, label_added2 = False, False
# ax22.spines['bottom'].set_visible(False)

n = 1

for i, (bf, gen) in enumerate(zip(feasibledf.BestFit, feasibledf.Generation)):
    non_improvement_counter += 1

    if bf < minbf:
        scatdict={"marker":"|",'s':500, 'zorder':20, "fc":"red", "ec":"red"}
        minbf=bf
        if not label_added1:
            ax11.scatter(gen,bf, label="Improvement", **scatdict)
            label_added1 = True
        else:
            scatdict['s'], scatdict['zorder']=100, 20
            ax22.scatter(gen, bf, **scatdict)
            ax101.scatter(gen, bf, **scatdict)
        non_improvement_counter = 0
    # elif non_improvement_counter == 20:
    #
    #     if not label_added2:
    #         ax11.scatter(gen,bf,s=500, zorder=500, marker="|", fc="yellow", label="20 gen's w/o improvement")
    #         label_added2 = True
    #     else:
    #         ax11.scatter(gen, bf, s=500, zorder=500, marker="|", fc="yellow")
    #     string = str(non_improvement_counter) + " gens w/o improvement"
    #     non_improvement_counter = 0
    #
    #     ax11.annotate(bf, (gen,bf), (-30, 100), textcoords="offset points", ha="left",
    #                  va="bottom",rotation=45, color="yellow",
    #                  # backgroundcolor="white",
    #                  arrowprops=dict(arrowstyle="->", color="yellow"),
    #                  label="20 generations w/o improvement",
    #                  zorder=100, fontsize=12)

# l = ax11.get_xlim()
# l2 = ax33.get_xlim()
# f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
# ticks = f(ax11.get_xticks())
# ax33.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))


maxgen = "Gen: " + str(tinydf.Generation.iloc[-1]) + "\nFitness: " + ("%.5g" % tinydf.BestFit.iloc[-1])

# legend = ax11.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=3, framealpha=0)
# ax22.get_legend().remove()

for i, ax in enumerate([ax11,ax22,ax33, ax44, ax100,ax101, ax111]):
    if i <2:
        ax.scatter(tinydf.Generation.iloc[-1],tinydf.BestFit.iloc[-1],zorder=200,
                   fc="green", s=100, marker="x")
        ax.annotate(maxgen, (tinydf.Generation.iloc[-1],tinydf.BestFit.iloc[-1]),(0,50),
                    textcoords="offset points", arrowprops=dict(arrowstyle="-"),ha="center",backgroundcolor="white", color="black")
    ax.set_xlim(left=0)
    # ax.get_legend().remove()
    defaultAxisSettings(ax,axes="both", floorx=True)
    # ax.xaxis.set_tick_params(which='both', labelbottom=True)
# ax33.xaxis.set_tick_params(which='both', labelleft=False)
# let's draw a line between both graphs indicating the first generation with a fitness below 1
# X, Y = belowone['Generation'].iloc[0], belowone['BestFit'].iloc[0]

# transFigure = fig1.transFigure.inverted()
# coord1 = transFigure.transform(ax11.transData.transform([X,Y]))
# coord2 = transFigure.transform(ax22.transData.transform([X,Y]))
#
# line = mpl.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),
#                                transform=fig1.transFigure, color="green",zorder=1,linestyle="--")
# fig1.lines = line,







# ax11.legend(loc="upper right").set_zorder(30)

# ax22.set_ylim(top=1)

# for i, chunk in enumerate(d['genomes']):
#     rows_end += chunk
#     new_dataframe['generation'].append(i)
#     new_df = df[columns_to_read].loc[rows_start:rows_end]

    # statistics of the data
    # new_df.drop(new_df[new_df['fitness'] >= 2.0].index, inplace=True) # remove all infeasible results
    # for key in ['max_fitness', 'min_fitness', 'mean_fitness', 'std_fitness']:
    #     if "max" in key:
    #         val = new_df['fitness'].max()
    #     elif "min" in key:
    #         val = new_df['fitness'].min()
    #     elif "std" in key:
    #         val = new_df['fitness'].std()
    #     elif "mean" in key:
    #         val = new_df['fitness'].mean()
    #     new_dataframe[key].append(val)
    #
    # for key in ['max_cost', 'min_cost', 'mean_cost', 'std_cost']:
    #     if "max" in key:
    #         val = new_df['cost'].max()
    #     elif "min" in key:
    #         val = new_df['cost'].min()
    #     elif "std" in key:
    #         val = new_df['cost'].std()
    #     elif "mean" in key:
    #         val = new_df['cost'].mean()
    #     new_dataframe[key].append(val)
    #
    # for key in ['max_footprint', 'min_footprint', 'mean_footprint', 'std_footprint']:
    #     if "max" in key:
    #         val = new_df['footprint'].max()
    #     elif "min" in key:
    #         val = new_df['footprint'].min()
    #     elif "std" in key:
    #         val = new_df['footprint'].std()
    #     elif "mean" in key:
    #         val = new_df['footprint'].mean()
    #     new_dataframe[key].append(val)

    # for key in ['max_mass', 'min_mass', 'mean_mass', 'std_mass']:
    #     if "max" in key:
    #         val = new_df['mass'].max()
    #     elif "min" in key:
    #         val = new_df['mass'].min()
    #     elif "std" in key:
    #         val = new_df['mass'].std()
    #     elif "mean" in key:
    #         val = new_df['mass'].mean()
    #     new_dataframe[key].append(val)

    #
    #     # find the row closest to the median, necessary of plotting actual solutions and
    #     # not averaged/rounded off solutions
    #     f = new_df['fitness'].values.tolist() # fitnesses as list
    #     median_value = f[np.argmin(abs(f - np.median(f)))]
    #     if max(f) < 10:

    #     a=(a.values.tolist())

    #     for key in columns_to_read:

    #
    #         new_dataframe[key].append(a[key].values[0])
    # rows_start = rows_end + 1


# ga_results = pd.DataFrame(new_dataframe)  # the final dataframe




# fig2, axs = plt.subplots(N, sharex=True)

# for i, ax in enumerate(axs):
#     what = columns_to_read[i]  # returns e.g. fitness, cost etc...
#
#     ax.plot(ga_results['generation'], ga_results[('mean_' + what)], '-r', label="$\mu$")
#     ax.scatter(ga_results['generation'].max(), ga_results[('mean_' + what)].iloc[-1], marker="x", ec="red",
#                zorder=30)  #
#     ax.annotate(str(ga_results['generation'].max()),
#                 xy=(ga_results['generation'].max(), ga_results[('mean_' + what)].iloc[-1] * 1.1), xycoords='data',
#                 horizontalalignment='center', verticalalignment='bottom', zorder=40)
#     #
#     ax.annotate(str(ga_results[('mean_' + what)].iloc[-1]),
#                 xy=(ga_results['generation'].max(), ga_results[('mean_' + what)].iloc[-1]), xycoords='data',
#                 xytext=(0.8, 0.95), textcoords='axes fraction',
#                 arrowprops=dict(facecolor='black', shrink=0.05),
#                 horizontalalignment='right', verticalalignment='top',
#                 )
#     # #
#     ax.fill_between(ga_results['generation'], ga_results[('min_' + what)], ga_results[('max_' + what)],
#                     ec=(0, 0.502, 0.502, 0.8), fc=(0, 0.502, 0.502, 0.3), zorder=1, label="min/max")
#     ax.fill_between(ga_results['generation'], ga_results[('mean_' + what)] + ga_results[('std_' + what)],
#                     ga_results[('mean_' + what)] - ga_results[('std_' + what)],
#                     ec=(0, 0.502, 0.502, 1), fc=(0, 0.502, 0.502, 0.9), zorder=2, label="$\sigma$")
#     #
#     ax.minorticks_on()
#     ax.grid(b=True, which='both', color='#999999', linestyle='-', alpha=0.2, axis="both")
#     ax.grid(b=True, which='both', color='#999999', linestyle='-', alpha=0.2, axis="both")
#
#     if not "Fitness" in columns_to_read[i]:
#         # ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.2)
#         ax.set_xlim(left=0)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     # add arrows
#     # get width and height of axes object to compute
#     # matching arrowhead length and width
#     xmin, xmax = ax.get_xlim()
#     ymin, ymax = ax.get_ylim()
#
# l = axs[0].legend(loc="upper right", frameon=False)
# l.set_zorder(20)
#
# # add labels
# axs[0].set_ylabel('Fitness')
# axs[1].set_ylabel('Cost ($€$)')
# axs[2].set_ylabel('Footprint ($kgCO_2e$)')
# # axs[3].set_ylabel('Mass ($kg$)')
# axs[-1].set_xlabel('Generations')  # shared xlabel
#
# every_nth = 1
# for n, label in enumerate(axs[-1].xaxis.get_ticklabels()):
#     if n % every_nth != 0:
#         label.set_visible(False)
# plt.show()
#

# fig1, (ax10) = plt.subplots(1, sharex=True)
# # ax10.plot(ga_results['generation'],ga_results['min_mass'], color=(1,0,0,1), label="mass")
# # ax10.fill_between(ga_results['generation'],ga_results['min_mass'], [0]*len(ga_results['min_mass']), ec = (1, 0, 0, 1), fc = (1, 0, 0, 0.2), zorder=1)
#
# ax10.plot([-5,-5],[-5,-5],color=(0,0.5,0.7,1), label="cost")  # dummy to create label
# ax10.plot(ga_results['generation'],ga_results['min_footprint'], color=(0,0,1,1), label="footprint")
#

# # twin object for two different y-axis on the sample plot
# ax33=ax10.twinx()
# # make a plot with different y-axis using second axis object
# # ax33.plot(gapminder_us.year, gapminder_us["gdpPercap"],color="blue",marker="o")
# ax33.plot(ga_results['generation'],ga_results['min_cost'], color=(0,0.5,0.7,1), label="cost")
# ax10.set_ylabel("Mass ($kg$), Footprint ($kgCO_2e$)")
# ax33.set_ylabel("Cost ($€$)")
# ax10.set_xlabel('Generations')
# l = ax10.legend(loc="upper right", frameon=False)
# l.set_zorder(20)

# plt.tight_layout()

# fig1.savefig(r"G:\My Drive\Thesis\Møter\test.png", bbox_inches='tight', transparent=True)
plt.show()

