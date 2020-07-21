import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_errbar(data,year=2019,err='MAE'):
    # plot data['Error']
    data['Year']   = data['DateTime'].apply(lambda x: x.year)
    data['Month']  = data['DateTime'].apply(lambda x: x.month)
    m_name = ['J','F','M','A','M','J','J','A','S','O','N','D']
    fig, ax = plt.subplots(1,1,figsize=(16, 10),)
    box_plot = data[data['Year'] ==2019]
    sns.boxplot(x="Month", y="{}".format(err),data=box_plot, width=0.8,linewidth=1, ax=ax)
    ax.tick_params(axis="x", labelsize=20)
    xk = data['Month'].unique() -1
    ax.set_xticks(xk)
    ax.set_xticklabels(m_name)
    ax.set_xlabel("Year {}".format(year), fontsize=20)
    ax.set_ylabel("{} ".format(err), fontsize=24)
    ax.tick_params(axis="y", labelsize=20)
    #ax.legend(title="Northwind",title_fontsize=20,fontsize=20)
    fig.text(0.5, 0.02, 'Time', ha='center',size=24)
    plt.show()

#read file
def read_error_file(types):
    data_path = 'C:/0_Academy/VScode/01_Data/01_ws_model/'
    file_name = 'Error01.xlsx'
    df   = pd.read_excel(data_path + file_name, sheet_name='{}'.format(types))
    observations = df.copy()
    #observations['DateTime'] = pd.to_datetime(observations['DateTime'])
    print("Import Done")
    return observations

def Plot_Error_step123(sheet,metric='MAPE',ylim=[0,40],colors='pastel'):
    data =  read_error_file(sheet)
    step =['1st-step','2nd-step','3rd-step']
    fig, ax = plt.subplots(1,len(step),sharey=True, figsize=(16,6), gridspec_kw = {'wspace':0, 'hspace':0})
    for i in range(len(step)):
        data01 = data[data['Step']==i+1]
        sns.barplot(x="Season", y=metric,hue="Method",data=data01,linewidth=1, ax=ax[i]
                    , palette=colors)
        ax[i].tick_params(axis="x", labelsize=14)
        ax[i].set_xlabel("{} Prediction".format(step[i]), fontsize=16)
    ax[0].set_ylim(ylim)
    ax[0].set_ylabel(metric, fontsize=24)
    ax[0].tick_params(axis="y", labelsize=20)
    ax[0].legend(loc=2,title=sheet,title_fontsize=18,fontsize=18)
    for j in range(len(step)-1):
        ax[j+1].get_legend().remove()
        ax[j+1].set_ylabel('')
    fig.text(0.5, -0.05, 'N-step prediction', ha='center',size=24)
    fig.text(0.5, 0.9, "N-step prediction error ({})".format(metric), ha='center',size=24)
    plt.show()

def plot_errbar_Season(data,year=2019,err='MAE'):
    fig, ax = plt.subplots(1,1,figsize=(10,6))
    colors = ["mediumspringgreen",'salmon','khaki',"paleturquoise"]
    seasons=['Spring','Summer','Autumn','Winter']
    plot = data[data['Year'] ==2019]
    sns.set_palette(sns.color_palette(colors))
    sns.violinplot(x="Season", y="{}".format(err),hue="Seasons",hue_order=seasons,data=plot,
                   linewidth=3,ax=ax,dodge=False)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("Season", fontsize=20)
    ax.set_ylabel("Error".format(), fontsize=20)
    ax.set_ylim(-10,10)
    ax.set_yticks([-10,-7.5,-5,-2.5,0,2.5,5,7.5,10])
    #ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(seasons)
    ax.set_title("First-step prediction Error in Each Seasons", fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    #ax.legend(loc=2,title="Season",title_fontsize=24,fontsize=24,fancybox=True, framealpha=0.5)
    ax.get_legend().remove()
    plt.show()
