"""Plot measured train-only validation scores, without inspecting test results."""
from pathlib import Path
import os, json
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(root/'.tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
report=json.loads((root/'reports'/'metrics.json').read_text())
rows=report['training']['candidates']
labels=[('Word' if item['features']=='word' else 'Word + character')+f'\nC = {item["C"]}' for item in rows]
fig,ax=plt.subplots(figsize=(10,4.5),facecolor='#F8F4E7')
ax.set_facecolor('#F8F4E7')
bars=ax.bar(labels,[item['f1']*100 for item in rows],color=['#4E342E']*4+['#CC5500'],width=.6)
ax.set_ylim(88,92); ax.set_ylabel('Positive-class F1 (%)'); ax.set_title('Model selection on 5,000 validation reviews\nOfficial test labels were not used for selection',color='#4E342E',pad=16)
for bar,item in zip(bars,rows):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.1,f'{item["f1"]*100:.3f}%',ha='center',color='#4E342E',fontsize=11)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig(root/'reports'/'validation_search.png',dpi=160)
print('Saved reports/validation_search.png from actual validation measurements')
