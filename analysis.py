# -*- coding: utf-8 -*-
"""
电商美妆类目用户画像与经营分析 —— 一键复现脚本
数据：Olist Brazilian E-Commerce Public Dataset（Kaggle 开源）
依赖：pandas, numpy, matplotlib
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), 'data')
OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 读取 ----------
orders = pd.read_csv(f'{D}/olist_orders_dataset.csv',
                     parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date',
                                  'order_estimated_delivery_date'])
items = pd.read_csv(f'{D}/olist_order_items_dataset.csv')
reviews = pd.read_csv(f'{D}/olist_order_reviews_dataset.csv')
cust = pd.read_csv(f'{D}/olist_customers_dataset.csv')
prod = pd.read_csv(f'{D}/olist_products_dataset.csv')
tr = pd.read_csv(f'{D}/product_category_name_translation.csv')

prod = prod.merge(tr, on='product_category_name', how='left')
prod['cat'] = prod['product_category_name_english'].fillna(prod['product_category_name'])

# ---------- 2. 明细宽表（仅已交付订单） ----------
base = (orders[orders.order_status == 'delivered']
        [['order_id', 'customer_id', 'order_purchase_timestamp',
          'order_delivered_customer_date', 'order_estimated_delivery_date']]
        .merge(items[['order_id', 'product_id', 'price']], on='order_id')
        .merge(prod[['product_id', 'cat']], on='product_id'))
base['month'] = base.order_purchase_timestamp.dt.to_period('M')
base['beauty'] = base.cat.isin(['health_beauty', 'perfumery'])

beauty = base[base.beauty]
print(f"已交付订单 {base.order_id.nunique():,} | 总GMV R${base.price.sum():,.0f}")
print(f"美妆GMV R${beauty.price.sum():,.0f} | 占比 {100*beauty.price.sum()/base.price.sum():.1f}%")

# ---------- 3. 趋势 ----------
trend = base.groupby(['month', 'beauty']).price.sum().unstack().loc['2017-01':'2018-08']
share = 100 * trend[True] / (trend[True] + trend[False])

# ---------- 4. 复购与用户画像 ----------
bo = beauty.groupby(['order_id', 'customer_id', 'order_purchase_timestamp']) \
           .price.sum().reset_index().rename(columns={'order_purchase_timestamp': 'ts'})
cust_order = bo.merge(cust[['customer_id', 'customer_unique_id', 'customer_state']], on='customer_id')

rep = lambda df: (df.groupby('customer_unique_id').order_id.nunique() > 1).mean()
all_orders = base.groupby(['order_id', 'customer_id']).size().reset_index() \
                 .merge(cust[['customer_id', 'customer_unique_id']], on='customer_id')
print(f"平台复购率 {100*rep(all_orders):.1f}% | 美妆复购率 {100*rep(cust_order):.1f}%")

# ---------- 5. RFM 分层 ----------
snap = cust_order.ts.max() + pd.Timedelta(days=1)
rfm = cust_order.groupby('customer_unique_id').agg(
    R=('ts', lambda x: (snap - x.max()).days),
    F=('order_id', 'nunique'), M=('price', 'sum'))
rfm['R分'] = pd.qcut(rfm.R, 4, labels=[4, 3, 2, 1]).astype(int)
rfm['M分'] = pd.qcut(rfm.M.rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['分层'] = np.select(
    [(rfm['R分'] >= 3) & (rfm['M分'] >= 3),
     (rfm['R分'] >= 3) & (rfm['M分'] < 3),
     (rfm['R分'] < 3) & (rfm['M分'] >= 3)],
    ['高价值活跃', '活跃低消费', '沉睡高价值'], '流失风险')
seg = rfm.groupby('分层').agg(用户数=('M', 'size'), 人均GMV=('M', 'mean'), GMV=('M', 'sum'))
seg['GMV占比%'] = 100 * seg.GMV / seg.GMV.sum()
seg['用户占比%'] = 100 * seg.用户数 / seg.用户数.sum()
print(seg.round(1))

# ---------- 6. 价格带 ----------
bins = [0, 50, 100, 150, 200, 300, 500, 10000]
labels = ['<50', '50-100', '100-150', '150-200', '200-300', '300-500', '500+']
pb = beauty.assign(band=pd.cut(beauty.price, bins, labels=labels)) \
           .groupby('band', observed=True).price.agg(['size', 'sum'])
pb['GMV占比%'] = 100 * pb['sum'] / pb['sum'].sum()
print(pb.round(1))

# ---------- 7. 口碑与履约 ----------
rv = reviews.groupby('order_id').review_score.first()
od = orders[orders.order_status == 'delivered'][
    ['order_id', 'order_delivered_customer_date', 'order_estimated_delivery_date']].copy()
od['delay'] = (od.order_delivered_customer_date - od.order_estimated_delivery_date).dt.days
od['score'] = od.order_id.map(rv)
od['beauty'] = od.order_id.isin(set(beauty.order_id))
od['delay_bin'] = pd.cut(od.delay, [-400, 0, 3, 7, 14, 400],
                         labels=['提前/准时', '晚1-3天', '晚4-7天', '晚8-14天', '晚14天+'])
dd = od.groupby(['delay_bin', 'beauty'], observed=True).score.mean().unstack()
dd.columns = ['平台其他', '美妆']
print(dd.round(2))

# ---------- 7.5 产品维度：哪些产品受喜爱，哪些在砸口碑 ----------
b_items = beauty.merge(od[['order_id', 'score']].drop_duplicates(), on='order_id', how='left')
prod_stat = b_items.groupby('product_id').agg(
    销量=('price', 'size'), GMV=('price', 'sum'), 均分=('score', 'mean'),
    差评率=('score', lambda x: 100 * (x <= 2).mean()), 类目=('cat', 'first')).reset_index()
prod_stat = prod_stat[prod_stat.销量 >= 5]          # 过滤低销量噪声
q75 = prod_stat.销量.quantile(.75)                   # 热销线：销量前 25%
loved = prod_stat[(prod_stat.销量 >= q75) & (prod_stat.均分 >= 4.3)].sort_values('GMV', ascending=False)
hated = prod_stat[(prod_stat.销量 >= q75) & (prod_stat.均分 <= 3.8)].sort_values('差评率', ascending=False)
print(f"热销产品 {int((prod_stat.销量 >= q75).sum())} 个 | 受喜爱 {len(loved)} 个 | 问题产品 {len(hated)} 个")
print(f"受喜爱品均差评率 {loved.差评率.mean():.1f}% | 问题品均差评率 {hated.差评率.mean():.1f}%")

# 连带分析：多单率与纯美妆订单占比
multi = (b_items.groupby('order_id').product_id.nunique() > 1).mean()
bo_cats = base[base.order_id.isin(set(beauty.order_id))].groupby('order_id').cat.apply(set)
only_b = bo_cats.apply(lambda s: s <= {'health_beauty', 'perfumery'}).mean()
print(f"美妆多单率 {100*multi:.1f}% | 纯美妆订单占比 {100*only_b:.1f}%")

# ---------- 8. 图表 ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0, 0]
tm = trend.index.to_timestamp()
ax.plot(tm, trend[False].values, label='全平台(除美妆)', color='#9ecae1', lw=1.8)
ax.plot(tm, trend[True].values, label='美妆类目', color='#d45087', lw=2.5)
ax.set_title('月度GMV趋势：美妆 vs 平台其他类目'); ax.legend(loc='upper left'); ax.grid(alpha=.3)
ax2 = ax.twinx(); ax2.plot(tm, share.values, color='#888', ls='--', alpha=.6)
ax2.set_ylabel('美妆GMV占比 %'); ax2.set_ylim(0, 30)

ax = axes[0, 1]
s = seg.loc[['高价值活跃', '沉睡高价值', '活跃低消费', '流失风险']]
x = np.arange(len(s)); w = .38
ax.bar(x - w/2, s['用户占比%'], w, label='用户占比%', color='#9ecae1')
ax.bar(x + w/2, s['GMV占比%'], w, label='GMV占比%', color='#d45087')
ax.set_xticks(x); ax.set_xticklabels(s.index)
ax.set_title('美妆买家RFM分层：用户 vs GMV贡献'); ax.legend(); ax.grid(alpha=.3, axis='y')

ax = axes[1, 0]
ax.bar(pb.index.astype(str), pb['GMV占比%'].round(1), color='#d45087', alpha=.85)
ax.set_title('美妆价格带GMV分布（R$）'); ax.set_ylabel('GMV占比 %'); ax.grid(alpha=.3, axis='y')

ax = axes[1, 1]
dd.plot(kind='bar', ax=ax, color=['#9ecae1', '#d45087'])
ax.set_title('交付延迟天数 vs 评论均分'); ax.set_ylabel('平均评分'); ax.set_xlabel('')
plt.setp(ax.get_xticklabels(), rotation=20); ax.axhline(3, color='gray', ls=':', alpha=.5)
ax.legend(); ax.grid(alpha=.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/dashboard.png', dpi=110)
print('图表已输出到 figures/dashboard.png')

# 产品维度四象限图
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
ax.scatter(prod_stat.销量, prod_stat.均分, s=prod_stat.GMV / 60, alpha=.5,
           c='#d45087', edgecolors='white', lw=.4)
ax.axhline(4.3, color='#2b8a3e', ls='--', lw=1)
ax.axhline(3.8, color='#c92a2a', ls='--', lw=1)
ax.axvline(q75, color='gray', ls=':', lw=1)
ax.text(q75 + 2, 4.55, '受喜爱区', color='#2b8a3e', fontsize=11)
ax.text(q75 + 2, 3.2, '问题产品区', color='#c92a2a', fontsize=11)
ax.set_xlabel('销量（件）'); ax.set_ylabel('平均评分')
ax.set_title('热销美妆产品：销量 × 评分（气泡=GMV）'); ax.grid(alpha=.3)
ax = axes[1]
bars = ax.bar(['受喜爱热销品\n(差评地板≈物流)', '问题热销品\n(产品本身问题)'],
              [loved.差评率.mean(), hated.差评率.mean()], color=['#9ecae1', '#d45087'])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{b.get_height():.1f}%', ha='center')
ax.set_ylabel('平均差评率 %'); ax.set_title('差评率对比：地板 vs 真问题'); ax.grid(alpha=.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/product_analysis.png', dpi=110)
print('图表已输出到 figures/product_analysis.png')
