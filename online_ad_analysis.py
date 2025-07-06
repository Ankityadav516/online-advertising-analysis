import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("online_advertising_performance_data.csv")

# Clean numeric data
numeric_cols = ['displays', 'cost', 'clicks', 'revenue', 'post_click_conversions', 'post_click_sales_amount']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Clean categorical columns
categorical_cols = ['campaign_number', 'user_engagement', 'banner', 'placement']
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.strip())

# Create date column
df['date'] = pd.to_datetime(df['month'] + ' ' + df['day'].astype(str) + ', 2020', format='%B %d, %Y')
df.drop(['month', 'day'], axis=1, inplace=True)

# Add derived metrics
df['conversion_rate'] = df['post_click_conversions'] / df['clicks'].replace(0, np.nan)
df['roi'] = df['revenue'] / df['cost'].replace(0, np.nan)
df['cpc'] = df['cost'] / df['clicks'].replace(0, np.nan)
df['cost_per_conversion'] = df['cost'] / df['post_click_conversions'].replace(0, np.nan)
df['day_of_week'] = df['date'].dt.day_name()

# ----- Save Visualizations -----

# 1. Engagement Trend Over Time
engagement_trend = df.groupby(['date', 'user_engagement']).size().unstack().fillna(0)
engagement_trend.plot(title="User Engagement Over Time", figsize=(12, 5))
plt.ylabel("Engagement Count")
plt.tight_layout()
plt.savefig("images/engagement_trend.png")
plt.clf()

# 2. Clicks by Banner Size
sns.boxplot(x='banner', y='clicks', data=df)
plt.title("Clicks by Banner Size")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/clicks_by_banner.png")
plt.clf()

# 3. Sales Trend Over Time
sales_trend = df.groupby('date')['post_click_sales_amount'].sum()
sales_trend.plot(title="Post-Click Sales Over Time", figsize=(12, 5))
plt.ylabel("Sales Amount ($)")
plt.tight_layout()
plt.savefig("images/sales_trend.png")
plt.clf()

# 4. Engagement by Banner
sns.countplot(x='banner', hue='user_engagement', data=df)
plt.title("User Engagement by Banner Size")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/engagement_by_banner.png")
plt.clf()

# 5. Displays & Clicks Over Time
df.groupby('date')['displays'].sum().plot(label='Displays')
df.groupby('date')['clicks'].sum().plot(label='Clicks')
plt.title("Displays and Clicks Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("images/displays_clicks_trend.png")
plt.clf()

# 6. Post-Click Conversions by Placement
sns.boxplot(x='placement', y='post_click_conversions', data=df)
plt.title("Post-Click Conversions by Placement")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/conversions_by_placement.png")
plt.clf()

# 7. User Engagement by Day of Week
df.groupby(['day_of_week', 'user_engagement']).size().unstack().plot(kind='bar', figsize=(12, 5))
plt.title("User Engagement by Day of Week")
plt.tight_layout()
plt.savefig("images/engagement_by_weekday.png")
plt.clf()

# 8. Cost Per Click (CPC) by Campaign & Banner
cpc_trend = df.groupby(['campaign_number', 'banner'])['cpc'].mean().unstack()
cpc_trend.plot(kind='bar', figsize=(14, 5))
plt.title("Cost Per Click by Campaign and Banner")
plt.tight_layout()
plt.savefig("images/cpc_by_campaign_banner.png")
plt.clf()

# 9. Conversion Rate by Day of Week
df.groupby('day_of_week')['conversion_rate'].mean().plot(kind='bar')
plt.title("Conversion Rate by Day of Week")
plt.tight_layout()
plt.savefig("images/conversion_by_day.png")
plt.clf()

# 10. Conversion Rate by User Engagement
df.groupby('user_engagement')['conversion_rate'].mean().plot(kind='bar')
plt.title("Conversion Rate by User Engagement")
plt.tight_layout()
plt.savefig("images/conversion_by_engagement.png")
plt.clf()

# ----- Print Summary Stats -----

# Top 5 Campaigns by Conversion Rate
top_campaigns = df.groupby('campaign_number')['conversion_rate'].mean().sort_values(ascending=False)
print("Top Campaigns by Conversion Rate:\n", top_campaigns.head(5))

# Top Placements by Displays
top_placements = df.groupby('placement').agg({'displays': 'sum', 'clicks': 'sum'}).sort_values('displays', ascending=False)
print("Top Placements by Displays:\n", top_placements.head(5))

# Revenue per Engagement
engagement_revenue = df.groupby('user_engagement')['revenue'].sum()
print("Revenue by User Engagement:\n", engagement_revenue)

# Cost-Effective Campaigns
cost_effective = df.groupby(['campaign_number', 'placement'])['cost_per_conversion'].mean().sort_values()
print("Most Cost-Effective Campaigns and Placements:\n", cost_effective.head(5))

# ROI by Campaign & Banner
top_roi = df.groupby(['campaign_number', 'banner'])['roi'].mean().sort_values(ascending=False)
print("Top Campaigns and Banners by ROI:\n", top_roi.head(5))

# Correlation: Cost vs Revenue
corr = df[['cost', 'revenue']].corr(method='spearman').iloc[0, 1]
print(f"Correlation (Cost vs Revenue): {corr:.2f}")

# Average Revenue Per Click
avg_rev_per_click = df['revenue'].sum() / df['clicks'].sum()
print(f"Average Revenue per Click: ${avg_rev_per_click:.2f}")
