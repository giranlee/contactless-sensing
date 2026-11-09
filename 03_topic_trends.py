import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

plt.rcParams['font.family'] = 'DejaVu Sans'
DPI = 300
INK, MUTED, GRID = '#262626', '#5f5f5c', '#e6e6e3'

LABEL = {0: 'RFID-based indoor location and care management', 1: 'Radar-based gait assessment', 2: 'Contactless sleep and vital sign monitoring', 3: 'Deep learning–based radar fall detection', 4: 'Radar-based human activity recognition', 5: 'Radar-based fall and human motion classification', 6: 'Wi-Fi CSI–based fall and activity detection'}
DOMAIN = {0: 'Location and activity monitoring', 4: 'Location and activity monitoring', 1: 'Physiological and functional assessment', 2: 'Physiological and functional assessment', 3: 'Fall detection', 5: 'Fall detection', 6: 'Fall detection'}
DCOL = {'Location and activity monitoring': '#2a78d6', 'Physiological and functional assessment': '#eb6834', 'Fall detection': '#1baf7a'}
EARLY_END = 2014

def clean_axes(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#bdbdb8')
    ax.tick_params(colors=INK, labelsize=9)

def jmir_p(p):
    if p < .001: return 'P<.001'
    if p < .01: return 'P=' + f'{p:.3f}'.lstrip('0')
    r2 = round(p, 2)
    if r2 >= 1: return 'P>.99'
    if (p < .05) != (r2 < .05): return 'P=' + f'{p:.3f}'.lstrip('0')
    return 'P=' + f'{r2:.2f}'.lstrip('0')

NUM = {0: 1, 4: 2, 1: 3, 2: 4, 3: 5, 5: 6, 6: 7}

doc = pd.read_excel('bertopic_2_doc_topics.xlsx')
doc = doc[doc.Topic != -1].copy()
early_x = doc.loc[doc['Year'] <= EARLY_END, 'Year'].mean()
doc['Period'] = np.where(doc['Year'] <= EARLY_END, early_x, doc['Year'])
ct = pd.crosstab(doc['Period'], doc['Topic']).sort_index()
prop = ct.div(ct.sum(axis=1), axis=0) * 100
x = ct.index.values.astype(float)

rows = []
for t in prop.columns:
    y = prop[t].values
    r = stats.linregress(x, y)
    resid = y - (r.intercept + r.slope * x)
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)
    trend = ('hot' if r.slope > 0 else 'cold') if r.pvalue < .05 else ('warm' if r.slope > 0 else 'cool')
    rows.append({'Topic': t, 'Topic no.': f'T{NUM[t]}', 'Label': LABEL[t], 'Articles, n': int(ct[t].sum()), 'B (%/year)': round(r.slope, 2), 'SE': round(r.stderr, 2), 'β': round(r.rvalue, 3), 't': round(r.slope / r.stderr, 2), 'df': len(x) - 2, 'P value': r.pvalue, 'P (JMIR)': jmir_p(r.pvalue), 'Durbin-Watson': round(dw, 2), 'Trend': trend})
reg = pd.DataFrame(rows)
reg = reg.sort_values('Topic', key=lambda c: c.map(NUM)).reset_index(drop=True)
xlab = [f'2006–{EARLY_END}'] + [str(int(v)) for v in x[1:]]
def trend_table(df, early, denom_all=False):
    d = df.copy()
    ex = d.loc[d['Year'] <= early, 'Year'].mean() if (d['Year'] <= early).any() else None
    d['P'] = np.where(d['Year'] <= early, ex, d['Year'])
    c = pd.crosstab(d['P'], d['Topic'])
    den = c.sum(axis=1) if denom_all else c.drop(columns=[-1], errors='ignore').sum(axis=1)
    c = c.drop(columns=[-1], errors='ignore')
    p = c.div(den, axis=0) * 100; xx = p.index.values.astype(float); out = {}
    for t in p.columns:
        r = stats.linregress(xx, p[t].values)
        tr = ('hot' if r.slope > 0 else 'cold') if r.pvalue < .05 else ('warm' if r.slope > 0 else 'cool')
        ptxt = jmir_p(r.pvalue)
        out[t] = f'{tr} (B={r.slope:.2f}, {ptxt})'.replace('=-', '=−')
    return out
all_doc = pd.read_excel('bertopic_2_doc_topics.xlsx')
asg = all_doc[all_doc.Topic != -1]
sens = pd.DataFrame({'Main: 2006–2014 combined, topic-assigned articles': trend_table(asg, EARLY_END), 'S1: 2026 (partial year) excluded': trend_table(asg[asg['Year'] < 2026], EARLY_END), 'S2: 2015–2026 only (annual)': trend_table(asg[asg['Year'] >= 2015], 0), 'S3: denominator = all articles (incl. outliers)': trend_table(all_doc, EARLY_END, denom_all=True), 'S4: 2006–2016 combined': trend_table(asg, 2016)})
sens = sens.loc[sorted(sens.index, key=NUM.get)]
sens.index = [f'T{NUM[t]}. {LABEL[t]}' for t in sens.index]

with pd.ExcelWriter('Table6_regression.xlsx') as w:
    reg.to_excel(w, sheet_name='regression', index=False)
    sens.to_excel(w, sheet_name='sensitivity')
    out = ct.copy(); out.index = xlab; out.to_excel(w, sheet_name='articles by period')
    out = prop.round(1); out.index = xlab; out.to_excel(w, sheet_name='proportion (%)')

fig, axes = plt.subplots(2, 4, figsize=(13, 7.0), dpi=DPI, sharey=True)
ORDER = [0, 4, 1, 2, 3, 5, 6]
for ax, (_, r) in zip(axes.flat, reg.set_index('Topic', drop=False).loc[ORDER].iterrows()):
    t = r.Topic; c = DCOL[DOMAIN[t]]
    ax.plot(x, prop[t].values, color=c, lw=2, marker='o', ms=4, zorder=3)
    fit = stats.linregress(x, prop[t].values)
    ax.plot(x, fit.intercept + fit.slope * x, color=INK, lw=1, ls='--', zorder=2)
    ax.set_title(textwrap.fill(f'T{NUM[t]}. {LABEL[t]}', 34), fontsize=9, color=INK, loc='left')
    ptxt = jmir_p(r['P value'])
    ax.text(0.97, 0.93, f'{r.Trend} (B={r["B (%/year)"]:.2f}, {ptxt})'.replace('=-', '=−'), transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color=INK, fontweight='bold' if r.Trend in ('hot', 'cold') else 'normal')
    ax.set_yticks([0, 20, 40, 60]); ax.set_ylim(-12, 75)
    ax.set_xticks([x[0], 2015, 2018, 2021, 2024, 2026]); ax.set_xticklabels([xlab[0], '2015', '2018', '2021', '2024', '2026'], fontsize=7.5, rotation=45)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); clean_axes(ax)
axes.flat[-1].axis('off')
axes.flat[-1].legend(handles=[Line2D([], [], color=c, lw=2, label=d) for d, c in DCOL.items()] +
                     [Line2D([], [], color=INK, lw=1, ls='--', label='Linear trend')], loc='center', frameon=False, fontsize=9)
fig.supylabel('Proportion of topic-assigned articles (%)', fontsize=10, color=INK)
plt.tight_layout(); plt.savefig('Figure4_topic_trends.png', facecolor='white'); plt.close()

print('saved: Figure4_topic_trends.png, Table6_regression.xlsx')
print(reg.to_string(index=False))
