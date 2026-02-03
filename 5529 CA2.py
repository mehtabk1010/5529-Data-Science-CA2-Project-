# ============================================
# DATA SCIENCE PROJECT - COFFEE SALES ANALYSIS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load data
df = pd.read_csv("C:\\Users\\khanm\\Downloads\\Coffe_sales.csv")

print("=" * 60)
print("COFFEE SALES DATA SCIENCE PROJECT")
print("=" * 60)

# ============================================
# MANDATORY COMMANDS 
# ============================================

print("\n📋 MANDATORY COMMANDS (4 commands)")
print("-" * 40)

# 1. head() command
print("\n1. head() - First 5 rows:")
print(df.head())
print()

# 2. info() command
print("2. info() - Dataset information:")
df.info()
print()

# 3. describe() command
print("3. describe() - Statistical summary:")
print(df.describe())
print()

# 4. shape command
print("4. shape - Dataset dimensions:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()

# ============================================
# TECHNIQUE 1-5: DATA CLEANING
# ============================================

print("\n🧹 DATA CLEANING TECHNIQUES (5 techniques)")
print("-" * 40)

# Technique 1: Check for missing values
print("\n🔹 TECHNIQUE 1: Check missing values")
missing = df.isnull().sum()
print("Missing values:")
print(missing)
print()

# Technique 2: Check for duplicates
print("🔹 TECHNIQUE 2: Check duplicate rows")
duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
print()

# Technique 3: Remove duplicates
print("🔹 TECHNIQUE 3: Remove duplicates")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")
else:
    print("No duplicates to remove")
print(f"Rows after cleaning: {df.shape[0]}")
print()

# Technique 4: Standardize column names (Renaming)
print("🔹 TECHNIQUE 4: Standardize column names (Renaming)")
print("Original columns:", df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("Cleaned columns:", df.columns.tolist())
print()

# Technique 5: Fix data types
print("🔹 TECHNIQUE 5: Fix data types")
print("Before fixing - Data types:")
print(df.dtypes)

# Fix date column
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

# Fix money column if needed
if df['money'].dtype == 'object':
    df['money'] = pd.to_numeric(df['money'], errors='coerce')

print("\nAfter fixing - Data types:")
print(df.dtypes)
print()

# ============================================
# TECHNIQUE 6-9: DATA TRANSFORMATION
# ============================================

print("\n🔄 DATA TRANSFORMATION TECHNIQUES (4 techniques)")
print("-" * 40)

# Technique 6: Extract date features
print("\n🔹 TECHNIQUE 6: Extract date features")
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.day_name()
print("Added: year, month, day, day_of_week")
print()

# Technique 7: Extract time features
print("🔹 TECHNIQUE 7: Extract time features")
# Clean time column
df['time'] = df['time'].astype(str).str.split('.').str[0]
df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
print(f"Extracted hour from time (range: {df['hour'].min()}-{df['hour'].max()})")
print()

# Technique 8: Create time categories
print("🔹 TECHNIQUE 8: Create time categories")
def time_category(hour):
    if hour < 12:
        return 'Morning'
    elif hour < 17:
        return 'Afternoon'
    else:
        return 'Night'

df['time_category'] = df['hour'].apply(time_category)
print("Created: Morning, Afternoon, Night categories")
print()

# Technique 9: Binary encoding
print("🔹 TECHNIQUE 9: Binary encoding")
df['is_card'] = df['cash_type'].apply(lambda x: 1 if x == 'card' else 0)
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x in ['Sat', 'Sun'] else 0)
print("Created binary columns: is_card, is_weekend")
print()

# ============================================
# TECHNIQUE 10-13: FEATURE ENGINEERING
# ============================================

print("\n⚙️ FEATURE ENGINEERING TECHNIQUES (4 techniques)")
print("-" * 40)

# Technique 10: Create price categories
print("\n🔹 TECHNIQUE 10: Create price categories")
def price_category(price):
    if price < 25:
        return 'Low'
    elif price < 35:
        return 'Medium'
    else:
        return 'High'

df['price_category'] = df['money'].apply(price_category)
print("Price categories: Low, Medium, High")
print()

# Technique 11: Create coffee groups
print("🔹 TECHNIQUE 11: Create coffee groups")
def coffee_group(name):
    name = str(name).lower()
    if 'latte' in name:
        return 'Latte'
    elif 'cappuccino' in name:
        return 'Cappuccino'
    elif 'americano' in name:
        return 'Americano'
    elif 'espresso' in name:
        return 'Espresso'
    elif 'chocolate' in name or 'cocoa' in name:
        return 'Chocolate'
    else:
        return 'Other'

df['coffee_group'] = df['coffee_name'].apply(coffee_group)
print("Coffee groups created")
print()

# Technique 12: Create peak hour indicator
print("🔹 TECHNIQUE 12: Create peak hour indicator")
peak_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in peak_hours else 0)
print(f"Peak hours: {peak_hours}")
print()

# Technique 13: Create daily totals
print("🔹 TECHNIQUE 13: Create daily totals")
daily_sales = df.groupby('date')['money'].sum().reset_index()
daily_sales.columns = ['date', 'daily_total']
df = df.merge(daily_sales, on='date', how='left')
print("Added daily total sales")
print()

# ============================================
# TECHNIQUE 14-17: EXPLORATORY DATA ANALYSIS
# ============================================

print("\n📊 EXPLORATORY DATA ANALYSIS TECHNIQUES (4 techniques)")
print("-" * 40)

# Technique 14: Sales distribution histogram
print("\n🔹 TECHNIQUE 14: Sales distribution (Histogram)")
plt.figure(figsize=(10, 5))
plt.hist(df['money'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sales Amount')
plt.xlabel('Sales Amount ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
print()

# Technique 15: Popular coffee types bar chart
print("🔹 TECHNIQUE 15: Popular coffee types (Bar Chart)")
top_coffees = df['coffee_name'].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_coffees.plot(kind='bar', color='lightgreen')
plt.title('Top 10 Most Popular Coffee Types')
plt.xlabel('Coffee Type')
plt.ylabel('Number of Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print()

# Technique 16: Sales by day line chart
print("🔹 TECHNIQUE 16: Sales by day (Line Chart)")
daily_trend = df.groupby('date')['money'].sum()
plt.figure(figsize=(12, 5))
daily_trend.plot(kind='line', color='blue')
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print()

# Technique 17: Correlation heatmap
print("🔹 TECHNIQUE 17: Correlation analysis (Heatmap)")
numeric_cols = ['money', 'hour', 'is_card', 'is_weekend', 'is_peak_hour']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
print()

# ============================================
# TECHNIQUE 18-20: TEXT DATA TECHNIQUES
# ============================================

print("\n📝 TEXT DATA TECHNIQUES (3 techniques)")
print("-" * 40)

# Technique 18: Text length analysis
print("\n🔹 TECHNIQUE 18: Coffee name length analysis")
df['name_length'] = df['coffee_name'].astype(str).str.len()

plt.figure(figsize=(10, 5))
plt.hist(df['name_length'], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.title('Coffee Name Length Distribution')
plt.xlabel('Name Length (characters)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
print()

# Technique 19: Word frequency analysis
print("🔹 TECHNIQUE 19: Most common words in coffee names:")
all_words = ' '.join(df['coffee_name'].astype(str).str.lower()).split()
word_counts = Counter(all_words)

print("Top 10 words:")
top_words = word_counts.most_common(10)
for word, count in top_words:
    print(f"   {word:15s} : {count:4d} times")
print()

print("Top 10 most common words in coffee names:")
top_words = word_counts.most_common(10)
for word, count in top_words:
    print(f"  '{word}': {count} times")
print()

# Technique 20: One-hot encoding
print("🔹 TECHNIQUE 20: One-hot encoding of coffee groups")
dummies = pd.get_dummies(df['coffee_group'], prefix='coffee')
df = pd.concat([df, dummies], axis=1)
print(f"Created {dummies.shape[1]} dummy variables")
print("Dummy columns:", list(dummies.columns))
print()

# ============================================
# OUTLIER HANDLING (REQUIRED - not counted in 20)
# ============================================

print("\n⚠️ OUTLIER HANDLING (Required - not in 20 techniques)")
print("-" * 40)

print("Outlier detection using IQR method:")
Q1 = df['money'].quantile(0.25)
Q3 = df['money'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['money'] < lower_bound) | (df['money'] > upper_bound)]
print(f"Lower bound: ${lower_bound:.2f}")
print(f"Upper bound: ${upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)}")
print(f"Percentage: {(len(outliers)/len(df))*100:.2f}%")
print()

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("📈 PROJECT COMPLETION SUMMARY")
print("="*60)

# Calculate final statistics
total_sales = df['money'].sum()
avg_sale = df['money'].mean()
total_transactions = len(df)
most_popular = df['coffee_name'].value_counts().index[0]
card_pct = (df['is_card'].sum() / len(df)) * 100

print(f"""
✅ MANDATORY COMMANDS COMPLETED: 4
   1. head()    - View first rows
   2. info()    - Dataset information  
   3. describe()- Statistical summary
   4. shape     - Dataset dimensions

✅ 20 TECHNIQUES COMPLETED:

🧹 DATA CLEANING (5 techniques):
   1. Check missing values
   2. Check duplicate rows  
   3. Remove duplicates
   4. Standardize column names ✓ (Renaming done)
   5. Fix data types

🔄 DATA TRANSFORMATION (4 techniques):
   6. Extract date features
   7. Extract time features
   8. Create time categories
   9. Binary encoding

⚙️ FEATURE ENGINEERING (4 techniques):
   10. Create price categories
   11. Create coffee groups
   12. Create peak hour indicator
   13. Create daily totals

📊 EXPLORATORY DATA ANALYSIS (4 techniques):
   14. Sales distribution (Histogram)
   15. Popular coffee types (Bar Chart)
   16. Sales by day (Line Chart)
   17. Correlation analysis (Heatmap)

📝 TEXT DATA ANALYSIS (3 techniques):
   18. Text length analysis
   19. Word frequency analysis
   20. One-hot encoding

⚠️ ADDITIONAL REQUIREMENTS COMPLETED:
   • Outlier handling using IQR method
   • Inconsistent renaming of columns (Technique 4)

📊 KEY FINDINGS:
   • Total transactions: {total_transactions:,}
   • Total revenue: ${total_sales:,.2f}
   • Average sale: ${avg_sale:.2f}
   • Most popular coffee: {most_popular}
   • Card payments: {card_pct:.1f}%
   • Date range: {df['date'].min().date()} to {df['date'].max().date()}
   • Missing values: {missing.sum()}
   • Duplicates removed: {duplicates}
   • Outliers detected: {len(outliers)}

💾 OUTPUT:
   • Processed file saved: 'coffee_sales_final.csv'
   • Final shape: {df.shape[0]} rows × {df.shape[1]} columns
""")

# Save final dataset
df.to_csv('coffee_sales_final.csv', index=False)
print("✅ Project completed successfully!")
print("="*60)
