# Tutorial 3 - Activity Answers

## Question: Walking Statistics
**What are the following statistics about?**

These statistics show the number of data points (rows) in each walking recording file. The describe() output gives:
- count: 31 walking files total
- mean: ~165 data points per file
- std: ~96 points variation
- min: 55 points (shortest walk)
- max: 446 points (longest walk)

Basically tells us walking sessions vary in length - some people walked longer than others.

---

## Activity 1 Questions

### 1. What are the activities included in the accelerometer dataset?

14 activities total:
- brush_teeth
- climb_stairs
- comb_hair
- descend_stairs
- drink_glass
- eat_meat
- eat_soup
- getup_bed
- liedown_bed
- pour_water
- sitdown_chair
- standup_chair
- use_telephone
- walk

### 2. Can you describe the characteristics of each activity based on the sample recordings?

**TODO:** Run cells 14-18 to generate plots, then observe:
- High-movement activities (walk, climb_stairs, descend_stairs): Large magnitude spikes, repetitive patterns
- Low-movement activities (sitdown_chair, liedown_bed, eat_soup): Smaller magnitude, less variation
- Medium activities (brush_teeth, pour_water, drink_glass): Moderate magnitude with specific motion patterns

Look at the X/Y/Z plots and magnitude plots to identify which axis shows most movement for each activity.

### 3. How do the accelerometer values differ across different activities? Are there any noticeable patterns or trends?

**TODO:** Check the plots from cells 14-18:
- Walking/stairs: Periodic oscillations (repetitive steps)
- Transitional movements (standup/sitdown): Sharp changes followed by stillness
- Static activities (eat_soup): Low variance, close to gravity baseline
- Tool-use activities (brush_teeth, comb_hair): Higher frequency, localized movements

Note which activities show clear periodicity vs random movements.

### 4. What insights can you gather from the statistical summaries of the accelerometer values for each activity?

**TODO:** Run cell 29 and cell 30, then compare:
- Mean values: Activities with consistent directions (e.g., liedown_bed might show Z-axis dominance)
- Standard deviation: Higher std = more dynamic movement (walk, climb_stairs)
- Min/Max ranges: Wider ranges indicate vigorous activities
- Compare magnitude statistics across activities - dynamic activities should have higher mean magnitude

### 5. Are there any significant differences in the distribution of accelerometer values between activities? If so, what are they?

**TODO:** Run cell 32 (distribution plots):
- Check if distributions are unimodal (one peak) or multimodal (multiple peaks)
- Static activities likely have narrow, peaked distributions
- Dynamic activities have wider, flatter distributions
- Some activities might show bimodal patterns (e.g., standup = initial push + standing still)

### 6. Can you identify any outliers or anomalies in the accelerometer data? How might these affect the analysis or interpretation of the results?

**TODO:** Create boxplots to identify outliers:
```python
plt.figure(figsize=(14, 6))
df.boxplot(column='Magnitude', by='Label', rot=45)
plt.title('Magnitude by Activity')
plt.show()
```

Look for:
- Points far outside whiskers = outliers
- Could be sensor errors, incorrect labels, or unusual volunteer behavior
- Impact: Outliers can skew feature calculations (mean, std) and hurt model performance
- Solution: Consider removing extreme outliers or using robust features (median, IQR)

### 7. How could the findings from the EDA be useful in developing an activity recognition model or application?

Practical uses:
- Feature selection: EDA shows magnitude is more stable than individual axes (rotation-invariant)
- Window sizing: Periodic activities need windows covering full cycles
- Class imbalance: If some activities have fewer samples, need to address (class weights, oversampling)
- Feature engineering: Activities with high variance need different features than static ones
- Filtering: High-frequency noise visible in plots suggests need for low-pass filter

### 8. How might the EDA findings be relevant in real-world scenarios, such as developing activity monitoring systems or fitness applications?

Real applications:
- **Health monitoring**: Detecting falls (sudden magnitude spike + orientation change)
- **Fitness tracking**: Counting steps from walk periodicity, estimating calories from activity intensity
- **Elderly care**: Detecting unusual patterns (e.g., no walking for extended periods)
- **Physical therapy**: Tracking specific movement quality over time
- **Context-aware apps**: Adjusting phone behavior based on current activity (e.g., silence during eating)

Key insight: Different activities need different processing approaches - can't use one-size-fits-all.

---

## Sliding Window Questions

### 1. What is the size of the original data (random_x)?

20 rows × 3 columns (20 time points, 3 axes)

### 2. What is the size of each window?

10 samples (window size = 10)

### 3. What is the shape of the result (win_x)?

**TODO:** Run cell 58 and print `win_x.shape`

Should be: (3, 10, 3)
- 3 windows
- 10 samples per window  
- 3 features (X, Y, Z axes)

### 4. What line is the beginning of the second window?

Line 6 (index 5)
- Window 1: lines 1-10
- Window 2: lines 6-15 (shifted by overlap=5)
- Window 3: lines 11-20

### 5. What is the meaning of the overlap parameter (ss)?

The step size - how many samples to shift forward for each new window.

In this example: ss=5 means 50% overlap
- More overlap = more windows = more training data but slower processing
- Less overlap = faster but might miss patterns between windows
- Common practice: 50% overlap balances data quantity and computational cost
This happened because you increased the Window Size (e.g., to 175).
I analyzed your data and found that 11 of your volunteer entries (like f1_1, f1_2, m1_1) have exactly 170 samples of data (approx 5.3 seconds).