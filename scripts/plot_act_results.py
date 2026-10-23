"""Regenerate the README figure with Python and matplotlib (optional dependency)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]

def main():
    before=list(csv.DictReader((ROOT/'docs/results/act-dagger7-screening.csv').open()))
    after=[r for r in csv.DictReader((ROOT/'docs/results/act-dagger36.csv').open()) if r['phase']=='screening']
    fig,ax=plt.subplots(figsize=(11,5.7),dpi=160)
    fig.patch.set_facecolor('#f7f9fc');ax.set_facecolor('#f7f9fc')
    fig.subplots_adjust(left=.085,right=.96,top=.77,bottom=.24)
    fig.text(.085,.935,'Learning to finish the placement',fontsize=22,weight='bold',color='#14243a')
    fig.text(.085,.875,'ACT checkpoint screening before and after 29 targeted corrections',fontsize=12,color='#45566e')
    for rows,color,label in [(before,'#7c8da5','7 correction segments'),(after,'#087e8b','36 correction segments')]:
        xs=[int(r['epoch']) for r in rows];ys=[100*int(r['successes'])/int(r['episodes']) for r in rows]
        ax.plot(xs,ys,color=color,label=label,lw=2.6,marker='o',ms=5)
    ax.scatter([90],[100*22/30],s=120,facecolor='#edac3e',edgecolor='#14243a',zorder=4)
    ax.annotate('Selected epoch 90\n22/30 screening successes',xy=(90,100*22/30),xytext=(56,90),fontsize=10,color='#14243a',arrowprops=dict(arrowstyle='-',color='#69788a'))
    ax.set(xlim=(5,105),ylim=(0,100),xticks=list(range(10,101,10)),xlabel='Training epoch',ylabel='Task success')
    ax.yaxis.set_major_formatter(PercentFormatter(100));ax.grid(axis='y',color='#dbe2ec',lw=.7);ax.set_axisbelow(True)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_color('#c0ccda')
    ax.tick_params(colors='#45566e');ax.legend(loc='lower right',frameon=False,fontsize=10)
    fig.text(.085,.13,'Fresh-seed confirmation of the selected model: 112/180 successes (62.2%)',fontsize=12,weight='bold',color='#14243a')
    fig.text(.085,.065,'Screening: seed 789 · 30 episodes/checkpoint · same architecture and training seed · execution horizon 16',fontsize=9,color='#45566e')
    out=ROOT/'docs/assets/act-checkpoint-success.png';out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    print(out)

if __name__=='__main__':main()
