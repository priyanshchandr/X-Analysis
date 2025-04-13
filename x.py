import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
file_path = "./Twitterdatainsheets.csv"
df = pd.read_csv(file_path, low_memory=False)

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Convert engagement columns to numeric, handling missing values
engagement_columns = ['RetweetCount', 'Likes']
df[engagement_columns] = df[engagement_columns].fillna(0).astype(int)

# Convert necessary columns to numeric
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
df['Sentiment'] = df['Sentiment'].astype(str).str.strip()
df['Weekday'] = df['Weekday'].astype(str).str.strip()

# Save processed dataset
df.to_csv("processed_tweet_data.csv", index=False)

# Generate summary statistics
summary_stats = df.describe()
summary_text = summary_stats.to_string()
    
# Save summary statistics to a text file
with open('summary_statistics.txt', 'w') as file:
    file.write(summary_text)

plt.figure(figsize=(8, 6))
sns.stripplot(data=df[['Likes', 'RetweetCount']], jitter=True, palette="coolwarm", alpha=0.5)
plt.yscale('log')
plt.xlabel("Engagement Type")
plt.ylabel("Count (Log Scale)")
plt.title("Strip Plot of Likes and Retweets (Scaled)")
plt.savefig("engagement_stripplot_scaled.png")
plt.close()

# Calculate total engagement (Likes + Retweets) for each user
df['TotalEngagement'] = df['Likes'] + df['RetweetCount']
top_users = df.groupby('UserID')['TotalEngagement'].sum().sort_values(ascending=False).head(10)

# Scatter Plot: Top Users by Total Engagement
plt.figure(figsize=(10, 6))
sns.scatterplot(x=top_users.index, y=top_users.values, s=100, color='purple', edgecolor='black', alpha=0.8)

plt.xlabel("User ID", fontsize=12)
plt.ylabel("Total Engagement (Likes + Retweets)", fontsize=12)
plt.title("Top 10 Users by Total Engagement", fontsize=14)

# Rotate X-axis labels if needed
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Display values above each point
for i, value in enumerate(top_users.values):
    plt.text(i, value + 5, str(value), ha='center', fontsize=10, color='black')

plt.grid(axis='y', linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("top_users_scatter.png")
plt.close()

# Visualization: Heatmap of Correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Visualization: Hourly Engagement Trends
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Hour'], y=df['Likes'], hue=df['Hour'], palette="coolwarm", legend=False)
plt.xlabel("Hour of the Day")
plt.ylabel("Likes")
plt.title("Likes Distribution by Hour")
plt.savefig("hourly_likes.png")
plt.close()


hourly_likes = df.groupby('Hour')['Likes'].mean()
hourly_likes.rolling(window=3).mean().plot(kind='line', figsize=(10, 6), marker='o', color='purple')
plt.xlabel("Hour of the Day")
plt.ylabel("Average Likes")
plt.title("Hourly Engagement Trend (Smoothed)")
plt.grid()
plt.savefig("hourly_engagement_trend.png")
plt.close()



# Visualization: Reach vs. Engagement Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Reach'], y=df['Likes'], alpha=0.5, color='purple')
plt.xlabel("Tweet Reach")
plt.ylabel("Likes")
plt.title("Tweet Reach vs Likes")
plt.savefig("reach_vs_likes.png")
plt.close()

# A/B Test: Comparing Engagement Between Morning (0-12) and Evening (12-23) Tweets
morning_tweets = df[df['Hour'] < 12]['Likes']
evening_tweets = df[df['Hour'] >= 12]['Likes']

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(morning_tweets, evening_tweets, equal_var=False)

# Save A/B test results
with open("ab_test_results.txt", "w") as f:
    f.write(f"A/B Test: Morning vs Evening Likes\n")
    f.write(f"T-Statistic: {t_stat}\n")
    f.write(f"P-Value: {p_value}\n")
    if p_value < 0.05:
        f.write("Statistically significant difference in engagement between morning and evening tweets.\n")
    else:
        f.write("No significant difference in engagement between morning and evening tweets.\n")

print("Processed dataset, summary statistics, graphs, and A/B test results saved successfully.")
