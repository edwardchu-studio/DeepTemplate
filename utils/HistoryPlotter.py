import imageio
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from PIL import GifImagePlugin
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
InteractiveShell.ast_node_interactivity='all'
plt.ion()

class HistoryPlotter:
    def __init__(self,):
        self.smooth_window=15
        self.chunk_threshold=1000
        pass
    def smooth(self,df,name):
        df[name]=df['Value'].rolling(self.smooth_window).mean()
        df['roll_max']=df[name]+df['Value'].rolling(self.smooth_window).std()
        df['roll_min']=df[name]-df['Value'].rolling(self.smooth_window).std()
        return df
    def chunk(self,df,condition):
        df=df.query(condition)
        return df
    def plot(self,data_dict,value_name,save=''):
        f=plt.figure()
        plt.title(value_name)
        for label,df in data_dict.items():
            if value_name=='W-Distance':
                df['Value']=-df['Value']
            df=self.smooth(df,value_name)
            df=self.chunk(df,'Step<={}'.format(self.chunk_threshold))
            ax=sns.lineplot(x='Step',y=value_name,data=df,label=label)
            plt.fill_between(df['Step'],y1=df['roll_min'],y2=df['roll_max'],alpha=0.3)
        ax.legend()
        if save:
            plt.savefig(save)
        plt.show()
pltr=HistoryPlotter()