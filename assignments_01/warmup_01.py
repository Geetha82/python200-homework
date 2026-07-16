import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#============ Pandas Review ==========
print(" \n ============ Pandas Review ========== \n")

#===== Pandas Question 1: Loading datasets and basic DataFrame exploration=====

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# ===== Print each result with a label =====
print(f"\n======Question 1: Shape and Data Types ======")
print("--- First Three Rows ---")
print(df.head(3))

print(f"\n--- DataFrame Shape ---")
print(f"Dimensions (Rows, Columns): {df.shape}")

print(f"\n--- Column Data Types ---")
print(df.dtypes)

#===== Pandas Question 2: Filtering on multiple conditions =====
print(f"\n======Question 2: Filtering on multiple conditions  ======")
# Filter for students where passed is True AND grade is strictly above 80
filtered_students = df[(df["passed"] == True) & (df["grade"] > 80)]

print("\n--- Filtered Students  ---")
print(f"Students Who Passed with a Grade Above 80:\n{filtered_students}")

#=====  Pandas Question 3: Adding New  Column - "grade-curved" =====
print(f"\n======Question 3: Adding New  Column - grade-curved  ======")
# Create a new column by adding 5 points to the existing grade column
df["grade_curved"] = df["grade"] + 5

print("\n--- Curved Grades Data ---")
print(f"Updated DataFrame with Curves Appended:\n{df}")

#=====  Pandas Question 4: Adding New  Column - "name_upper" =====
print(f"\n======Question 4: Adding New  Column - name_upper ======")
# create a new column with students name transformed to uppercase
df["name_upper"] = df["name"].str.upper()

print("\n--- Transformaed Names with Uppercase ---")
print(f"Updated DataFrame with uppercase names:\n{df}")

#=====  Pandas Question 5: Grouping and Aggregation =====
print(f"\n======Question 5: Grouping and Aggregation ======")
# Group by city and find the mean of the grade column
city_mean= df.groupby("city")["grade"].mean()

print(f"\n Mean Grade by City:\n{city_mean} ")

#=====  Pandas Question 6: Name Transformation =====
print(f"\n======Question 6: Name Transformation ======")
#Replace "Austin" with "Houston" inside the city column
df["city"] = df["city"].replace("Austin", "Houston")

print("\n--- City Value Replacement Confirmation ---")
print(f"Updated Name and City Columns:\n{df[['name', 'city']]}")

#=====  # Pandas Question 7: Sorting Records =====
print(f"\n======Question 7: Sorting Records ======")
#  Sort the DataFrame by the grade column in descending order
sorted_df = df.sort_values(by="grade", ascending=False)

print("\n--- Top Performing Students ---")
print(f"Top 3 Rows by Grade:\n{sorted_df.head(3)}")


#============ Numpy Review ==========
print(" \n ============ Numpy Review ========== \n")

#=====  Question 1: Array Attributes and Exploration =====
print(f"\n======Question 1: Array Attributes and Exploration  ======")
# Create a 1Dimensional NumPy array
number_list = [10, 20, 30, 40, 50]
number_arr = np.array(number_list)

print(f"Array Shape: {number_arr.shape}")
print(f"Data Type (dtype): {number_arr.dtype}")
print(f"Number of Dimensions (ndim): {number_arr.ndim}")

#=====  Question 2: Multi-dimensional Array Attributes =====
print(f"\n======Question 2: Multi-dimensional Array Attributes ======")
# Create the 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"2D Array Shape: {arr_2d.shape}")
print(f"Total Number of Elements (size): {arr_2d.size}")


#=====  Question 3: 2D Array Slicing =====
print(f"\n======Question 3: 2D Array Slicing ======")

# Slice out the rows from index 0 up to (but not including) 2, 
# and columns from index 0 up to (but not including) 2.
top_left_block = arr_2d[0:2, 0:2]

print(f"Top-Left 2x2 Block:\n{top_left_block}")

#=====  Question 4: Initialization (3, 4) and (2, 5) arrays =====
print(f"\n======Question 4: Initialization (3, 4) and (2, 5) arrays ======")

# create 3x4 array of zeros 
zeros_array = np.zeros((3, 4))

# create 2x5 array of ones 
ones_array = np.ones((2, 5))

# Print the arrays
print(f"3x4 Zeros Array:\n{zeros_array}")
print(f"\n2x5 Ones Array:\n{ones_array}")

#=====  Question 5: Arange and Statistical Methods =====
print(f"\n====== Question 5: Arange and Statistical Methods ======")

# Create the array with start=0, stop=50, step=5
range_array = np.arange(0, 50, 5)

# print the array, its shape, mean, sum, and standard deviation
print(f"Generated Array: {range_array}")
print(f"Array Shape: {range_array.shape}")
print(f"Mean: {range_array.mean()}")
print(f"Sum: {range_array.sum()}")
print(f"Standard Deviation: {range_array.std():.2f}")

#=====  Question 6: Random Value Arrays and Normal Distributions =====
print(f"\n====== Question 6: Random Value Arrays and Normal Distributions ======")

# Generate 200 values from a Normal (Gaussian) distribution
# Mean (loc) = 0, Standard Deviation (scale) = 1
random_vals = np.random.normal(loc=0, scale=1, size=200)

# Print the mean and standard deviation of the result
print(f"Sample Mean: {random_vals.mean():.4f}")
print(f"Sample Standard Deviation: {random_vals.std():.4f}")

#============ Matplotlib Review ==========
print(" \n ============ Matplotlib Review ========== \n")

#=====  Question 1: Line Plot =====
print(f"\n======Question 1: Line Plot  ======")

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Generate the plot
plt.plot(x, y)

# Add metadata labels
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

# Create the outputs folder if it doesn't exist yet
os.makedirs("assignments_01/outputs", exist_ok=True)

# SAVE the plot to  outputs folder BEFORE showing it
plt.savefig("assignments_01/outputs/lineplot.png")

# Display the plot
plt.show()


#=====  Question 2: Bar Plot =====
print(f"\n======Question 2: Bar Plot  ======")

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]


# Generate Bar plot
plt.bar(subjects, scores)

# Add metadata labels
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

# SAVE the plot to  outputs folder BEFORE showing it
plt.savefig("assignments_01/outputs/barchart.png")

# Display the plot
plt.show()


#=====  Question 3: Scatter Plot =====
print(f"\n======Question 3: Scatter Plot  ======")

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

# Plot the first dataset
plt.scatter(x1, y1, color='blue', label='Dataset 1')

# Plot the second dataset on the same figure
plt.scatter(x2, y2, color='red', label='Dataset 2')

# Add required metadata
plt.title("Scatter Plot Comparison")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")

# Add a legend to distinguish between the two colors
plt.legend()

# SAVE the plot to  outputs folder BEFORE showing it
plt.savefig("assignments_01/outputs/scatterplot.png")

# Display the plot
plt.show()

#=====  Question 4: Subplots (1 Row, 2 Columns) =====
print(f"\n====== Question 4: Subplots (1 Row, 2 Columns) ======")

# Define data from question 1 and 2
x_q1 = [0, 1, 2, 3, 4, 5]
y_q1 = [0, 1, 4, 9, 16, 25]
subjects_q2 = ["Math", "Science", "English", "History"]
scores_q2 = [88, 92, 75, 83]

# Create the figure and the 1x2 grid of axes (subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left Subplot: Line plot from Q1
ax1.plot(x_q1, y_q1, marker='o', color='blue')
ax1.set_title("Line Plot (Squares)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Right Subplot: Bar plot from Q2
ax2.bar(subjects_q2, scores_q2, color='skyblue')
ax2.set_title("Bar Plot (Subject Scores)")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

# Adjust layout to prevent overlapping labels
plt.tight_layout()

# SAVE the plot to  outputs folder BEFORE showing it
plt.savefig("assignments_01/outputs/subplots.png")

# Display the final combined figure
plt.show()

#============ Descriptive Statistics Review ==========
print(" \n ============ Descriptive Statistics Review ========== \n")

#=====  Question 1: Central Tendency and Dispersion =====
print(f"\n====== Question 1: Central Tendency and Dispersion ======")

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

# Convert to numpy array
data_arr = np.array(data)

# Compute statistics
data_mean = np.mean(data_arr)
data_median = np.median(data_arr)
data_variance = np.var(data_arr)
data_std = np.std(data_arr)

print(f"Mean: {data_mean}")
print(f"Median: {data_median}")
print(f"Variance: {data_variance:.2f}")
print(f"Standard Deviation: {data_std:.2f}")

#=====  Question 2: Normal Distribution Histogram =====
print(f"\n====== Question 2: Normal Distribution Histogram ======")

# Generate 500 random values: Mean=65, StdDev=10
random_scores = np.random.normal(65, 10, 500)

# Create the histogram
plt.hist(random_scores, bins=20, color='mediumseagreen', edgecolor='black')

# Add required metadata
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

# SAVE the plot to  outputs folder BEFORE showing it
plt.savefig("assignments_01/outputs/normal_distribution_histogram.png")

# Display the plot
plt.show()

#=====  Question 3: Boxplot Comparison =====
print(f"\n====== Question 3: Boxplot Comparison ======")

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

# Create the side-by-side boxplot
plt.boxplot([group_a, group_b], tick_labels =["Group A", "Group B"])

# Add required metadata
plt.title("Score Comparison")
plt.ylabel("Scores")

# 2. SAVE the plot to  outputs folder BEFORE showing it
plt.savefig("assignments_01/outputs/score_comparison_boxplot.png")

# Display the plot
plt.show()

#=====  Question 4: Distribution Comparison =====
print(f"\n====== Question 4: Distribution Comparison ======")

# Setup datasets
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Create the side-by-side boxplot
plt.boxplot([normal_data, skewed_data], tick_labels= ["Normal", "Exponential"])

# Add metadata
plt.title("Distribution Comparison")
plt.ylabel("Values")

# Save and Show
plt.savefig("assignments_01/outputs/distribution_comparison.png")
plt.show()

# 1. Which distribution is more skewed? 
#    The Exponential distribution is significantly more skewed (right-skewed).
#
# 2. Appropriate Measures of Central Tendency:
#    - For the Normal distribution, the MEAN is appropriate because the data is symmetric.
#    - For the Exponential distribution, the MEDIAN is more appropriate because the mean 
#      is easily pulled/inflated by the long tail of outliers.

#=====  Question 5: Central Tendency Comparison =====
print(f"\n====== Question 5: CEntral Tendency Comparison ======")

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

def get_stats(data, label):
    s = pd.Series(data)
    mean_val = s.mean()
    median_val = s.median()
    mode_val = s.mode()[0] # .mode() returns a Series, we take the first item
    
    print(f"\n{label}:")
    print(f"  Mean:   {mean_val}")
    print(f"  Median: {median_val}")
    print(f"  Mode:   {mode_val}")

get_stats(data1, "Data 1 (Symmetric-ish)")
get_stats(data2, "Data 2 (With Outlier)")

# Why are the median and mean so different for data2?
# The mean and median differ significantly in Data 2 because of the outlier value (150).
# The Mean is sensitive to every value and is "pulled" upward by the extreme high value.
# The Median only cares about the middle position, so it remains resistant to the outlier.

#============ Hypothesis Testing Review ==========
print(" \n ============ Hypothesis Testing Review ========== \n")

from scipy import stats

#=====  Question 1: Independent Samples T-Test =====
print(f"\n====== Question 1: Independent Samples T-Test ======")

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Run an independent samples t-test on the two groups
t_stat, p_value = stats.ttest_ind(group_a, group_b)

#  Print the t-statistic and p-value.
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_value:.8f}")

#=====  Question 2: Significance Interpretation =====
print(f"\n====== Question 2: Significance Interpretation ======")

# Define the significance level (alpha)
alpha = 0.05

# Evaluate the result
if p_value < alpha:
    print(f"Result is statistically significant (p < {alpha}). Reject the null hypothesis.")
else:
    print(f"Result is not statistically significant (p >= {alpha}). Fail to reject the null hypothesis.")


#=====  Question 3: Paired Samples T-Test =====
print(f"\n====== Question 3: Paired Samples T-Test ======")

# before and after scores (the same students measured twice)
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Run the paired t-test (ttest_rel stands for 'related' samples)
t_stat_pair, p_value_pair = stats.ttest_rel(before, after)

#  Print the t-statistic and p-value.
print(f"T-statistic: {t_stat_pair:.4f}")
print(f"P-value:     {p_value_pair:.8f}")

#=====  Question 4: One-Sample T-Test =====
print(f"\n====== Question 4: One-Sample T-Test ======")

scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

# Run the one-sample t-test against the benchmark
t_stat_onesample, p_value_onesample = stats.ttest_1samp(scores, benchmark)

print(f"T-statistic: {t_stat_onesample:.4f}")
print(f"P-value:     {p_value_onesample:.4f}")

#=====  Question 5: One-Tailed Independent T-Test =====
print(f"\n====== Question 5: One-Tailed Independent T-Test ======")

# Use the same groups from Question 1
# Run the test with alternative='less'
t_stat, p_val_one_tailed = stats.ttest_ind(group_a, group_b, alternative='less')

print(f"One-tailed P-value: {p_val_one_tailed:.10f}")

#=====  Question 6: Plain-Language Conclusion =====
print(f"\n====== Question 6: Plain-Language Conclusion ======")

print("\n--- Hypothesis Question 6: Conclusion ---")

conclusion = (
    "The analysis reveals a significant difference between the two groups. "
    "Students in Group B performed notably better than those in Group A, "
    "and given the extremely low p-value, it is highly unlikely that this "
    "result occurred by random chance."
)

print(conclusion)

#============ Correlation Review ==========
print(" \n ============ Correlation Review ========== \n")

#=====  Question 1: Pearson Correlation Matrix =====
print(f"\n====== Question 1: Pearson Correlation Matrix ======")

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Compute the Pearson correlation matrix
corr_matrix = np.corrcoef(x, y)

# Print the full correlation matrix
print(f"Full Correlation Matrix:\n{corr_matrix}")
# print just the correlation coefficient (the value at position [0, 1]).
print(f"Correlation Coefficient (r): {corr_matrix[0, 1]:.2f}")

# What do you expect the correlation to be, and why?
    # I expect the correlation to be exactly 1.0. 
    # This is because there is a perfect positive linear relationship between x and y 
    # (y is exactly 2 times x for every data point).


#=====  Question 2: Pearson Correlation with SciPy =====
print(f"\n====== Question 2: Pearson Correlation with SciPy ======")

from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# compute the correlation between x and y 
r_coeff, p_val = pearsonr(x, y)

# Print both the correlation coefficient and the p-value.
print(f"Correlation Coefficient (r): {r_coeff:.4f}")
print(f"P-value:                     {p_val:.8f}")

#=====  Question 3: Pandas Correlation Matrix =====
print(f"\n====== Question 3: Pandas Correlation Matrix ======")

# Define the dictionary 
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
# create dataframe
df = pd.DataFrame(people)

# compute the correlation matrix
people_corr_matrix = df.corr()

# Print the result.
print(f"People Correlation Matrix:\n{people_corr_matrix}")

#=====  Question 4: Visualizing Negative Correlation =====
print(f"\n====== Question 4: Visualizing Negative Correlation ======")

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

# Create a scatter plot of x and y which have a negative relationship. 
plt.scatter(x, y, color='red', marker='x')

# Add a title "Negative Correlation" and label both axes.
plt.title("Negative Correlation")
plt.xlabel("x values")
plt.ylabel("y values")

# Save and Show (keeping it in your outputs folder)
plt.savefig("assignments_01/outputs/negative_correlation_scatter.png")
plt.show()

print("Scatter plot 'Negative Correlation' displayed and saved successfully.")


#=====  Question 5: Heatmap Visualization =====
print(f"\n====== Question 5: Heatmap Visualization ======")

# 1. Use the correlation matrix from Question 3
# people_corr_matrix = df.corr()

# 2. Create the heatmap with numeric annotations (annot=True)
sns.heatmap(people_corr_matrix, annot=True, cmap='coolwarm')

# 3. Add required metadata
plt.title("Correlation Heatmap")

# 4. Save and Show
plt.savefig("assignments_01/outputs/correlation_heatmap.png")
plt.show()

print("Heatmap 'Correlation Heatmap' generated and saved successfully.")


#============ Pipelines ==========
print(" \n ============ Pipelines ========== \n")

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# define create_series(arr) 
def create_series(arr):
    """ takes a NumPy array and returns a pandas Series with the name "values"""
    return pd.Series(arr, name="values")

# define clean_data(series) 
def clean_data(series):
    """takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series"""
    return series.dropna()

# define summarize_data(series) 
def summarize_data(series):
    """takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode" """
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        # use series.mode()[0] to get a single value.
        "mode": series.mode()[0]
    }
    return summary

# data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary
def data_pipeline(arr):
    """Chains the processing steps in sequence."""
    # Step 1: Create
    series = create_series(arr)
    # Step 2: Clean
    cleaned_series = clean_data(series)
    # Step 3: Summarize
    results = summarize_data(cleaned_series)
    return results

# Call data_pipeline(arr) 
summary_results = data_pipeline(arr)

# print each key and its value from the result
for key, value in summary_results.items():
    print(f"{key.capitalize()}: {value:.2f}")

print("\n warmup_01.py is complete.")























