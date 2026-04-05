import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Setup ────────────────────────────────────────────────────────────────────
df = pd.read_csv('Loan_Default.csv')

C0 = "#2ecc71"   # Adimplente
C1 = "#e74c3c"   # Inadimplente
BG = "#f8f9fa"
ACCENT = "#2c3e50"
OUT = "graficos_correlacao/"

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

import os
os.makedirs(OUT, exist_ok=True)

# ─── Pearson Correlations ────────────────────────────────────────────────────
print("=== CORRELAÇÕES DE PEARSON ===")
pairs = [
    ('LTV', 'Status'),
    ('dtir1', 'Status'),
    ('Credit_Score', 'rate_of_interest'),
    ('loan_amount', 'Upfront_charges'),
    ('Credit_Score', 'Status'),
    ('rate_of_interest', 'Status'),
]
for a, b in pairs:
    d = df[[a, b]].dropna()
    r, p = stats.pearsonr(d[a], d[b])
    print(f"  {a} x {b}: r={r:.4f}, p={p:.2e}, n={len(d)}")

# ─── 1. LTV vs Status ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("LTV vs. Status de Inadimplência", fontsize=16, fontweight='bold', color=ACCENT)

d = df[df['LTV'] < 200].copy()
d['Status_label'] = d['Status'].map({0: 'Adimplente', 1: 'Inadimplente'})

bp = axes[0].boxplot(
    [d[d['Status']==0]['LTV'].dropna(), d[d['Status']==1]['LTV'].dropna()],
    labels=['Adimplente', 'Inadimplente'],
    patch_artist=True, medianprops=dict(color='black', linewidth=2)
)
bp['boxes'][0].set_facecolor(C0); bp['boxes'][0].set_alpha(0.8)
bp['boxes'][1].set_facecolor(C1); bp['boxes'][1].set_alpha(0.8)
axes[0].set_title('Distribuição do LTV por Status')
axes[0].set_ylabel('LTV (%)')

means = d.groupby('Status_label')['LTV'].mean()
bars = axes[1].bar(means.index, means.values, color=[C0, C1], alpha=0.85, width=0.5)
for bar, val in zip(bars, means.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('LTV Médio por Status')
axes[1].set_ylabel('LTV Médio (%)')
plt.tight_layout()
plt.savefig(OUT + '01_ltv_vs_status.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 2. DTI vs Status ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("DTI vs. Status de Inadimplência", fontsize=16, fontweight='bold', color=ACCENT)

d = df[['dtir1', 'Status']].dropna()
d['Status_label'] = d['Status'].map({0: 'Adimplente', 1: 'Inadimplente'})

bp = axes[0].boxplot(
    [d[d['Status']==0]['dtir1'], d[d['Status']==1]['dtir1']],
    labels=['Adimplente', 'Inadimplente'],
    patch_artist=True, medianprops=dict(color='black', linewidth=2)
)
bp['boxes'][0].set_facecolor(C0); bp['boxes'][0].set_alpha(0.8)
bp['boxes'][1].set_facecolor(C1); bp['boxes'][1].set_alpha(0.8)
axes[0].set_title('Distribuição do DTI por Status')
axes[0].set_ylabel('DTI (%)')

d['dti_bucket'] = pd.cut(d['dtir1'], bins=[0,20,30,40,50,62],
                          labels=['<20','20-30','30-40','40-50','>50'])
dr = d.groupby('dti_bucket', observed=True)['Status'].mean() * 100
axes[1].bar(dr.index.astype(str), dr.values, color='#e67e22', alpha=0.85, width=0.6)
for i, val in enumerate(dr.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('Taxa de Inadimplência por Faixa de DTI')
axes[1].set_ylabel('Taxa de Inadimplência (%)')
axes[1].set_xlabel('Faixa de DTI (%)')
plt.tight_layout()
plt.savefig(OUT + '02_dti_vs_status.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 3. Credit_Score vs rate_of_interest ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Credit Score vs. Taxa de Juros", fontsize=16, fontweight='bold', color=ACCENT)

d = df[['Credit_Score', 'rate_of_interest']].dropna()
sample = d.sample(min(5000, len(d)), random_state=42)
axes[0].scatter(sample['Credit_Score'], sample['rate_of_interest'],
                alpha=0.15, color='#3498db', s=10)
m, b, r, p, se = stats.linregress(d['Credit_Score'], d['rate_of_interest'])
x_line = np.linspace(d['Credit_Score'].min(), d['Credit_Score'].max(), 200)
axes[0].plot(x_line, m*x_line+b, color='#e74c3c', linewidth=2,
             label=f'r = {r:.3f}\np = {p:.3f}')
axes[0].set_xlabel('Credit Score'); axes[0].set_ylabel('Taxa de Juros (%)')
axes[0].set_title('Dispersão: Score vs. Juros'); axes[0].legend()

d['score_bucket'] = pd.cut(d['Credit_Score'], bins=[400,550,650,700,750,900],
                             labels=['400-550','550-650','650-700','700-750','750+'])
mean_rate = d.groupby('score_bucket', observed=True)['rate_of_interest'].mean()
axes[1].bar(mean_rate.index.astype(str), mean_rate.values, color='#9b59b6', alpha=0.85, width=0.6)
for i, val in enumerate(mean_rate.values):
    axes[1].text(i, val + 0.02, f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1].set_title('Juros Médio por Faixa de Score')
axes[1].set_ylabel('Taxa de Juros Média (%)')
axes[1].set_xlabel('Faixa de Credit Score')
plt.tight_layout()
plt.savefig(OUT + '03_creditscore_vs_interest.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 4. loan_amount vs Upfront_charges ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Valor do Empréstimo vs. Taxas Iniciais", fontsize=16, fontweight='bold', color=ACCENT)

d = df[['loan_amount', 'Upfront_charges']].dropna()
d = d[d['loan_amount'] < d['loan_amount'].quantile(0.99)]
d = d[d['Upfront_charges'] < d['Upfront_charges'].quantile(0.99)]
sample = d.sample(min(5000, len(d)), random_state=42)
axes[0].scatter(sample['loan_amount'], sample['Upfront_charges'], alpha=0.15, color='#1abc9c', s=10)
m, b, r, p, se = stats.linregress(d['loan_amount'], d['Upfront_charges'])
x_line = np.linspace(d['loan_amount'].min(), d['loan_amount'].max(), 200)
axes[0].plot(x_line, m*x_line+b, color='#e74c3c', linewidth=2, label=f'r = {r:.3f}')
axes[0].set_xlabel('Valor do Empréstimo (USD)'); axes[0].set_ylabel('Taxas Iniciais (USD)')
axes[0].set_title('Dispersão: Empréstimo vs. Taxas'); axes[0].legend()

d['charges_pct'] = (d['Upfront_charges'] / d['loan_amount']) * 100
d['loan_bucket'] = pd.cut(d['loan_amount'], bins=[0,100000,200000,300000,500000,10000000],
                            labels=['<100k','100-200k','200-300k','300-500k','>500k'])
mean_pct = d.groupby('loan_bucket', observed=True)['charges_pct'].mean()
axes[1].bar(mean_pct.index.astype(str), mean_pct.values, color='#1abc9c', alpha=0.85, width=0.6)
for i, val in enumerate(mean_pct.values):
    axes[1].text(i, val + 0.02, f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1].set_title('Taxa Inicial Média (% do Empréstimo)')
axes[1].set_xlabel('Faixa do Empréstimo'); axes[1].set_ylabel('Taxas Iniciais / Empréstimo (%)')
plt.tight_layout()
plt.savefig(OUT + '04_loanamount_vs_upfront.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 5. Region vs property_value ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Região vs. Valor do Imóvel", fontsize=16, fontweight='bold', color=ACCENT)

d = df[['Region','property_value']].dropna()
d = d[d['property_value'] < d['property_value'].quantile(0.99)]
region_order = d.groupby('Region')['property_value'].median().sort_values(ascending=False).index.tolist()
colors_r = ['#3498db','#e67e22','#9b59b6','#2ecc71']
data_by_region = [d[d['Region']==r]['property_value'].values for r in region_order]
bp = axes[0].boxplot(data_by_region, labels=region_order, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors_r): patch.set_facecolor(color); patch.set_alpha(0.8)
axes[0].set_title('Distribuição do Valor do Imóvel por Região')
axes[0].set_ylabel('Valor do Imóvel (USD)')
means_r = d.groupby('Region')['property_value'].median().reindex(region_order)
bars = axes[1].bar(means_r.index, means_r.values/1000, color=colors_r, alpha=0.85, width=0.6)
for bar, val in zip(bars, means_r.values/1000):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 f'${val:.0f}k', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1].set_title('Valor Mediano do Imóvel por Região')
axes[1].set_ylabel('Valor Mediano (USD mil)')
plt.tight_layout()
plt.savefig(OUT + '05_region_vs_propertyvalue.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 6. Region vs Status ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Região vs. Taxa de Inadimplência", fontsize=16, fontweight='bold', color=ACCENT)
dr = df.groupby('Region')['Status'].mean() * 100
dr = dr.sort_values(ascending=False)
bars = axes[0].bar(dr.index, dr.values, color=colors_r[:len(dr)], alpha=0.85, width=0.5)
for bar, val in zip(bars, dr.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[0].axhline(df['Status'].mean()*100, color='black', linestyle='--', linewidth=1.5,
                label=f'Média geral: {df["Status"].mean()*100:.1f}%')
axes[0].set_title('Taxa de Inadimplência por Região')
axes[0].set_ylabel('Taxa de Inadimplência (%)'); axes[0].legend()
count_d = df[df['Status']==1].groupby('Region').size()
count_t = df.groupby('Region').size()
stacked = pd.DataFrame({'Adimplentes': count_t - count_d, 'Inadimplentes': count_d}).reindex(dr.index)
stacked.plot(kind='bar', ax=axes[1], color=[C0, C1], alpha=0.85, width=0.6)
axes[1].set_title('Volume por Região e Status')
axes[1].set_xlabel('Região'); axes[1].set_ylabel('Número de Empréstimos')
axes[1].tick_params(axis='x', rotation=30); axes[1].legend()
plt.tight_layout()
plt.savefig(OUT + '06_region_vs_status.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 7. Occupancy vs LTV ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Tipo de Ocupação vs. LTV", fontsize=16, fontweight='bold', color=ACCENT)
d = df[['occupancy_type','LTV']].dropna()
d = d[d['LTV'] < 200]
occ_labels = {'pr': 'Residência Principal', 'ir': 'Investimento', 'sr': 'Segunda Residência'}
d['occ_label'] = d['occupancy_type'].map(occ_labels)
occ_order = ['Residência Principal', 'Investimento', 'Segunda Residência']
colors_o = ['#3498db','#e74c3c','#f39c12']
data_occ = [d[d['occ_label']==o]['LTV'].values for o in occ_order]
bp = axes[0].boxplot(data_occ, labels=occ_order, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors_o): patch.set_facecolor(color); patch.set_alpha(0.8)
axes[0].set_title('LTV por Tipo de Ocupação'); axes[0].set_ylabel('LTV (%)')
axes[0].tick_params(axis='x', rotation=15)
means_o = d.groupby('occ_label')['LTV'].mean().reindex(occ_order)
bars = axes[1].bar(means_o.index, means_o.values, color=colors_o, alpha=0.85, width=0.5)
for bar, val in zip(bars, means_o.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('LTV Médio por Tipo de Ocupação'); axes[1].set_ylabel('LTV Médio (%)')
axes[1].tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(OUT + '07_occupancy_vs_ltv.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 8. Age vs Income ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Faixa Etária vs. Renda", fontsize=16, fontweight='bold', color=ACCENT)
age_order = ['<25','25-34','35-44','45-54','55-64','65-74','>74']
d = df[['age','income']].dropna()
d = d[d['income'] < d['income'].quantile(0.99)]
d_filt = d[d['age'].isin(age_order)]
colors_age = plt.cm.viridis(np.linspace(0.2, 0.85, len(age_order)))
data_age = [d_filt[d_filt['age']==a]['income'].values for a in age_order]
bp = axes[0].boxplot(data_age, labels=age_order, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colors_age): patch.set_facecolor(color); patch.set_alpha(0.8)
axes[0].set_title('Distribuição da Renda por Faixa Etária'); axes[0].set_ylabel('Renda Mensal (USD)')
means_age = d_filt.groupby('age')['income'].median().reindex(age_order)
axes[1].plot(age_order, means_age.values, marker='o', color='#e74c3c', linewidth=2.5, markersize=8)
axes[1].fill_between(range(len(age_order)), means_age.values, alpha=0.15, color='#e74c3c')
axes[1].set_xticks(range(len(age_order))); axes[1].set_xticklabels(age_order)
axes[1].set_title('Renda Mediana por Faixa Etária'); axes[1].set_ylabel('Renda Mediana (USD)')
plt.tight_layout()
plt.savefig(OUT + '08_age_vs_income.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 9. Gender vs Credit_Score ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Gênero vs. Credit Score", fontsize=16, fontweight='bold', color=ACCENT)
d = df[df['Gender'].isin(['Male','Female','Joint'])][['Gender','Credit_Score']]
gender_colors = {'Male': '#3498db', 'Female': '#e91e8c', 'Joint': '#9b59b6'}
data_g = [d[d['Gender']==g]['Credit_Score'].values for g in ['Male','Female','Joint']]
bp = axes[0].boxplot(data_g, labels=['Masculino','Feminino','Conjunto'],
                     patch_artist=True, medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp['boxes'], gender_colors.values()): patch.set_facecolor(c); patch.set_alpha(0.8)
axes[0].set_title('Distribuição do Score por Gênero'); axes[0].set_ylabel('Credit Score')
means_g = d.replace({'Gender': {'Male':'Masculino','Female':'Feminino','Joint':'Conjunto'}}).groupby('Gender')['Credit_Score'].mean()
order_g = ['Masculino','Feminino','Conjunto']
means_g = means_g.reindex(order_g)
bars = axes[1].bar(means_g.index, means_g.values,
                   color=['#3498db','#e91e8c','#9b59b6'], alpha=0.85, width=0.5)
for bar, val in zip(bars, means_g.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('Score Médio por Gênero'); axes[1].set_ylabel('Credit Score Médio')
axes[1].set_ylim(600, 720)
plt.tight_layout()
plt.savefig(OUT + '09_gender_vs_creditscore.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 10. Age vs Status ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Faixa Etária vs. Inadimplência", fontsize=16, fontweight='bold', color=ACCENT)
d = df[df['age'].isin(age_order)][['age','Status']]
dr_age = d.groupby('age')['Status'].mean() * 100
dr_age = dr_age.reindex(age_order)
bars = axes[0].bar(age_order, dr_age.values,
                   color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(age_order))), alpha=0.85, width=0.6)
for bar, val in zip(bars, dr_age.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0].axhline(df['Status'].mean()*100, color='black', linestyle='--', linewidth=1.5,
                label=f'Média: {df["Status"].mean()*100:.1f}%')
axes[0].set_title('Taxa de Inadimplência por Faixa Etária')
axes[0].set_ylabel('Taxa de Inadimplência (%)'); axes[0].set_xlabel('Faixa Etária')
axes[0].legend()
count_age_d = d[d['Status']==1].groupby('age').size().reindex(age_order)
count_age_a = d[d['Status']==0].groupby('age').size().reindex(age_order)
stacked = pd.DataFrame({'Adimplentes': count_age_a, 'Inadimplentes': count_age_d})
stacked.plot(kind='bar', ax=axes[1], color=[C0, C1], alpha=0.85, width=0.7)
axes[1].set_title('Volume por Faixa Etária e Status')
axes[1].set_xlabel('Faixa Etária'); axes[1].set_ylabel('Número de Empréstimos')
axes[1].tick_params(axis='x', rotation=30); axes[1].legend()
plt.tight_layout()
plt.savefig(OUT + '10_age_vs_status.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 11. Interest-Only vs Status ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Modalidade Interest-Only vs. Inadimplência", fontsize=16, fontweight='bold', color=ACCENT)
d = df[['interest_only','Status']]
io_labels = {'not_int': 'Amortização Normal', 'int_only': 'Apenas Juros'}
d = d.copy(); d['io_label'] = d['interest_only'].map(io_labels)
dr_io = d.groupby('io_label')['Status'].mean() * 100
bars = axes[0].bar(dr_io.index, dr_io.values, color=['#2ecc71','#e74c3c'], alpha=0.85, width=0.5)
for bar, val in zip(bars, dr_io.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].set_title('Taxa de Inadimplência por Modalidade'); axes[0].set_ylabel('Taxa de Inadimplência (%)')
count_io = d.groupby(['io_label','Status']).size().unstack()
count_io.columns = ['Adimplente','Inadimplente']
count_io.plot(kind='bar', ax=axes[1], color=[C0, C1], alpha=0.85, width=0.5)
axes[1].set_title('Volume por Modalidade e Status')
axes[1].set_xlabel('Modalidade'); axes[1].set_ylabel('Número de Contratos')
axes[1].tick_params(axis='x', rotation=15); axes[1].legend()
plt.tight_layout()
plt.savefig(OUT + '11_interestonly_vs_status.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 12. Correlation Heatmap ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
fig.suptitle("Mapa de Calor — Correlação de Pearson", fontsize=15, fontweight='bold', color=ACCENT)

num_cols = ['loan_amount','rate_of_interest','Upfront_charges','property_value',
            'income','Credit_Score','LTV','dtir1','term','Status']
corr = df[num_cols].corr(method='pearson')
labels = {
    'loan_amount': 'Valor Empréstimo', 'rate_of_interest': 'Taxa de Juros',
    'Upfront_charges': 'Taxas Iniciais', 'property_value': 'Valor Imóvel',
    'income': 'Renda', 'Credit_Score': 'Credit Score',
    'LTV': 'LTV', 'dtir1': 'DTI', 'term': 'Prazo', 'Status': 'Status (Default)'
}
corr.rename(index=labels, columns=labels, inplace=True)
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 10, 'weight': 'bold'})
plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT + '12_heatmap_correlacao.png', dpi=150, bbox_inches='tight')
plt.close()

print("Todos os gráficos gerados com sucesso!")
