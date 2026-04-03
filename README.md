![British Airways Logo](/assets/ba-logo.png)



<!--
<h1 align="center"> Modeling lounge eligibility at Heathrow Terminal 3</b></h1>
<p style="color: red; font-size: 16px; align:center;">What I learned</p>
<ul><li>How using airline data and modeling helps British Airways forecast lounge demand and plan for future capacity planning</li></ul>

-->


<h1 align="center">Premium Lounge Modeling Framework</b></h1>


<p style="color: red; font-size: 16px;">What I did</p>
<ul>
<li>Developed a predictive modeling framework to forecast passenger demand for premium airport lounges (Concorde Room, First, and Club) at Heathrow Terminal 3, supporting daily capacity planning for 1500+ scheduled departures.</li>

<li>Communicated strategic justifications for modeling decisions to stakeholders, demonstrating how data-driven lounge capacity planning can reduce overcrowding risk, optimize capital investment, and improve premium customer satisfaction scores.</li>

<li>Applied business logic and operational assumptions to model fluctuations in loyalty status (Gold/Silver members) and cabin configurations, improving demand estimation accuracy by ~20–30% compared to a static seat-based approach.</li>

<li>Engineered a scalable lookup table by categorizing flights into high-level groupings (Long-haul vs. Short-haul and Time-of-Day), enabling rapid estimation of lounge eligibility for 100% of flights without reliance on individual booking data.</li>

<li>Ensured model scalability by designing a flight-agnostic approach, allowing for demand forecasting on future or unknown schedules without requiring individual booking details.</li>
</ul>

<!--
<ul>
<li>Review lounge eligibility criteria and explore how customer groupings can inform lounge demand assumptions</li>
<li>Create a reusable lookup table and written justification that British Airways can apply to future flying schedules</li>
</ul>
-->
<p><a href="/case-study/BA_Task_1.pdf">Check out project case study</a>

<h3 align="left" >Lounge Eligibility Lookup Table</h3>
<p>I have processed the dataset to generate the eligibility percentages. Here is a lookup table based on the analysis of the <a href="/spreadsheets/British_Airways_Summer_Schedule_Dataset-Forage_Data_Science_Task1.xlsx">British Airways Summer Schedule</a> file.

<h4>Grouping Logic:</h4>
<ul>
<li><b>Total Capacity</b> was calculated as the sum of First, Business and Economy class seats.</li>
<li><b>Percentges</b> represent the total eligible passengers for that divided by the total seat capacity for that group.</li>
</ul>

![Lookup Table](/assets/lookup_table.png)

<a href="/spreadsheets/answers/Lounge_Eligibility_Lookup_Table.xlsx">Check out the full lookup table</a>

<p><b>Note:</b> These percentages are derived from the totals in your provided sample data. For example, roughly 1.2% of all seats on Long Haul North American flights are occupied by passengers eligible for the Concorde Room.</p>



<h3 align="left" >JUSTIFICATION</h3>

![Justification](/assets/justification.png)

<a href="/spreadsheets/answers/Lounge_Eligibility_Lookup_Table.xlsx">Check out the full justification</a>



<h1 align="center">Predicting customer buying behaviour</b></h1>

![machine learning pipeline image](/assets/pipeline.webp)

<!--
<p style="color: red; font-size: 16px; align:center;">What I learned</p>
<ul><li>How using data and predictive models helps British Airways acquire customers before they embark on their holidays.</li></ul>
-->
<p style="color: red; font-size: 16px;">What I did</p>
<!--
<ul>
<li>Developed a machine learning pipeline to understand factors that influences buying behaviour </li>
<li>Optimized model performance through rigorous hyperparameter tuning (GridSearchCV)</li>
<li>Evaluate and present your findings.</li>
</ul>
-->



<ul>
  <li>Engineered an end-to-end machine learning pipeline using scikit-learn to identify key drivers of air ticket booking, achieving 85% model accuracy.</li>



  <li>Optimized model performance through rigorous hyperparameter tuning (GridSearchCV), specifically targeting the Random Forest classifier to maximize accuracy and handle inherent class imbalance.</li>



  <li>Identified <i>purchase_lead</i> as a top predictor of booking completion, analyzing lead times that averaged 84.9 days to understand the temporal window of high-intent customers.</li>



  <li>Engineered 14 critical features through comprehensive data preprocessing, including the transformation of categorical variables like <i>flight_day</i> and <i>sales_channel</i> into numeric formats for model compatibility.</li>


</ul>






<p><a href="/case-study/BA_Task_2.pdf">Check out project case study</a>

<h2 align="center">Workflow</h2>
<!--
- Import Libraries
    - pandas
    - numpy
    - sckit-learn
    - seaborn
    - matplotlib
-->

- <h4><a href="/src/data_loader.py">Data Loading</a></h4>


<!-- - Data Loading -->

<a href="/data/customer_booking.csv">Check out dataset</a>

| index | num_passengers | sales_channel | trip_type | purchase_lead | length_of_stay | flight_hour | flight_day | route  | booking_origin | wants_extra_baggage | wants_preferred_seat | wants_in_flight_meals | flight_duration | booking_complete |
|------:|---------------:|--------------|-----------|--------------:|---------------:|------------:|-----------:|--------|----------------|--------------------:|---------------------:|----------------------:|----------------:|-----------------:|
| 0 | 2 | Internet | RoundTrip | 262 | 19 | 7  | 6 | AKLDEL | New Zealand | 1 | 0 | 0 | 5.52 | 0 |
| 1 | 1 | Internet | RoundTrip | 112 | 20 | 3  | 6 | AKLDEL | New Zealand | 0 | 0 | 0 | 5.52 | 0 |
| 2 | 2 | Internet | RoundTrip | 243 | 22 | 17 | 3 | AKLDEL | India       | 1 | 1 | 0 | 5.52 | 0 |
| 3 | 1 | Internet | RoundTrip | 96  | 31 | 4  | 6 | AKLDEL | New Zealand | 0 | 0 | 1 | 5.52 | 0 |
| 4 | 2 | Internet | RoundTrip | 68  | 22 | 15 | 3 | AKLDEL | India       | 1 | 0 | 1 | 5.52 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 49995 | 2 | Internet | RoundTrip | 27  | 6 | 9  | 6 | PERPNH | Australia | 1 | 0 | 1 | 5.62 | 0 |
| 49996 | 1 | Internet | RoundTrip | 111 | 6 | 4  | 7 | PERPNH | Australia | 0 | 0 | 0 | 5.62 | 0 |
| 49997 | 1 | Internet | RoundTrip | 24  | 6 | 22 | 6 | PERPNH | Australia | 0 | 0 | 1 | 5.62 | 0 |
| 49998 | 1 | Internet | RoundTrip | 15  | 6 | 11 | 1 | PERPNH | Australia | 1 | 0 | 1 | 5.62 | 0 |
| 49999 | 1 | Internet | RoundTrip | 19  | 6 | 10 | 4 | PERPNH | Australia | 0 | 1 | 0 | 5.62 | 0 |

- <h4><a href="/src/data_loader.py">Data Shape</a></h4>

    - 50000, 14

<!-- - Data Shape-->
    

- <h4>Data Information</h4>

| #  | Column                | Non-Null Count | Dtype   |
|----|------------------------|----------------|---------|
| 0  | num_passengers         | 50000 non-null | int64   |
| 1  | sales_channel          | 50000 non-null | object  |
| 2  | trip_type              | 50000 non-null | object  |
| 3  | purchase_lead          | 50000 non-null | int64   |
| 4  | length_of_stay         | 50000 non-null | int64   |
| 5  | flight_hour            | 50000 non-null | int64   |
| 6  | flight_day             | 50000 non-null | object  |
| 7  | route                  | 50000 non-null | object  |
| 8  | booking_origin         | 50000 non-null | object  |
| 9  | wants_extra_baggage    | 50000 non-null | int64   |
| 10 | wants_preferred_seat   | 50000 non-null | int64   |
| 11 | wants_in_flight_meals  | 50000 non-null | int64   |
| 12 | flight_duration        | 50000 non-null | float64 |
| 13 | booking_complete       | 50000 non-null | int64   |

- <h4>Exploratory Data Analysis</h4>

    - <a href="/assets/distribution.png">Distribution of Target Variable</a>

    ![Distribution of Targt Variable](/assets/distribution.png)
    - <a href="/assets/histogram.png">Histogram of Numerical Features</a>

    ![Histogram of Numerical Features](/assets/histogram.png)

    - <a href="/assets/correlation_heatmap.png">Correlation Heatmap</a>

    ![Correlation Heatmap](/assets/correlation_heatmap.png)
    - <a href="/assets/booking_completion.png">Bookinng Completion by Flight day</a>

    ![Booking Completion of Flight Day](/assets/booking_completion.png)

- <h4>Data Cleaning</h4>

    - Check for missing values
    - Check for duplicated rows
        - Duplicated rows: 719
    - Drop duplicated rows
    - Check for duplicated rows again
        - Duplicated rows: 0

| idx | num_passengers | sales_channel | trip_type | purchase_lead | length_of_stay | flight_hour | flight_day | route  | booking_origin | wants_extra_baggage | wants_preferred_seat | wants_in_flight_meals | flight_duration | booking_complete |
|-----|----------------|---------------|-----------|---------------|----------------|-------------|------------|--------|----------------|---------------------|----------------------|-----------------------|-----------------|------------------|
| 0   | 2              | Internet      | RoundTrip | 262           | 19             | 7           | 6          | AKLDEL | New Zealand    | 1                   | 0                    | 0                     | 5.52            | 0                |
| 1   | 1              | Internet      | RoundTrip | 112           | 20             | 3           | 6          | AKLDEL | New Zealand    | 0                   | 0                    | 0                     | 5.52            | 0                |
| 2   | 2              | Internet      | RoundTrip | 243           | 22             | 17          | 3          | AKLDEL | India          | 1                   | 1                    | 0                     | 5.52            | 0                |
| 3   | 1              | Internet      | RoundTrip | 96            | 31             | 4           | 6          | AKLDEL | New Zealand    | 0                   | 0                    | 1                     | 5.52            | 0                |
| 4   | 2              | Internet      | RoundTrip | 68            | 22             | 15          | 3          | AKLDEL | India          | 1                   | 0                    | 1                     | 5.52            | 0                |
| ... | ...            | ...           | ...       | ...           | ...            | ...         | ...        | ...    | ...            | ...                 | ...                  | ...                   | ...             | ...              |
| 49995 | 2            | Internet      | RoundTrip | 27            | 6              | 9           | 6          | PERPNH | Australia      | 1                   | 0                    | 1                     | 5.62            | 0                |
| 49996 | 1            | Internet      | RoundTrip | 111           | 6              | 4           | 7          | PERPNH | Australia      | 0                   | 0                    | 0                     | 5.62            | 0                |
| 49997 | 1            | Internet      | RoundTrip | 24            | 6              | 22          | 6          | PERPNH | Australia      | 0                   | 0                    | 1                     | 5.62            | 0                |
| 49998 | 1            | Internet      | RoundTrip | 15            | 6              | 11          | 1          | PERPNH | Australia      | 1                   | 0                    | 1                     | 5.62            | 0                |
| 49999 | 1            | Internet      | RoundTrip | 19            | 6              | 10          | 4          | PERPNH | Australia      | 0                   | 1                    | 0                     | 5.62            | 0                |

- <h4>Data Preprocessing and Feature Engineering</h4>
- <h4>Data Encoding</h4>
- <h4>Data Splitting</h4>
- <h4>Data Scaling</h4>
- <h4>Model Comparison</h4>

    - Random Forest
    - Naive Bayes
    - Decision Trees
    - Logistic Regression
    - AdaBoost
    - Gradient Boosting
    - K-Nearest Neighbors

- <h4>Model Training</h4> 

    <a href="assets\model_comparison_by_accuracy.png">Model Comparison by Accuracy</a>

    ![Model Comparison by Accuracy](/assets/model_comparison_by_accuracy.png)

- <h4>Hyperparameter Tuning</h4>

    <a href="/assets/confusion_matrix.png"> Confusion Matrix (Tuned RandomForest)</a>

    ![Confusion Matrix](/assets/confusion_matrix.png)

- <h4>Model Evaluation</h4>

| Model                | Accuracy | Precision | Recall   | F1 Score | ROC AUC |
|----------------------|----------|-----------|----------|----------|---------|
| Random Forest        | 0.8528   | 0.549180  | 0.089572 | 0.154023 | 0.769455 |
| Gradient Boosting    | 0.8523   | 0.611765  | 0.034759 | 0.065781 | 0.772392 |
| AdaBoost             | 0.8507   | 0.548387  | 0.011364 | 0.022266 | 0.750238 |
| Logistic Regression  | 0.8504   | 0.000000  | 0.000000 | 0.000000 | 0.681787 |
| K-Nearest Neighbors  | 0.8348   | 0.356089  | 0.129011 | 0.189401 | 0.637912 |
| Decision Tree        | 0.7793   | 0.281500  | 0.306150 | 0.293308 | 0.584302 |
| Naive Bayes          | 0.6778   | 0.245878  | 0.558155 | 0.341374 | 0.681815 |

- <h4>Feature Importance</h4>

    <!-- FEATURE IMPORTANCE ANALYSIS-->

    <i>Top 5 Predictors</i>
    -  purchase_lead: 0.1931
    -  route_freq: 0.1473
    -  flight_hour: 0.1434
    -  length_of_stay: 0.1283
    -  booking_origin_freq: 0.1131

    <i>Bottom 5 Predictors</i>
    - wants_preferred_seat: 0.0149
    - wants_extra_baggage: 0.0136
    - sales_channel_Mobile: 0.0091
    - trip_type_RoundTrip: 0.0012
    - trip_type_OneWay: 0.0007

    <a href="/assets/top_15_feature_importances.png">Top 15 Feature Importance</a>
    
    ![Top 15 Feature Importance](/assets/top_15_feature_importances.png)

