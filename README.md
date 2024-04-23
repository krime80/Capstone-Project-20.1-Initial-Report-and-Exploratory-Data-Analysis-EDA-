# Capstone-Project-20.1-Initial-Report-and-Exploratory-Data-Analysis-EDA-
Creating and training models, as well as creating visualizations to make sense of the findings

**Author:** Krista Rime

#### Executive Summary
This project focuses on developing LSTM and GRU models to monitor event sequences critical to the Total Minutes Viewed ETL process. The models aim to detect and predict out-of-sequence events, significantly impacting partner and product reporting if left undetected. By leveraging these models, we can proactively identify potential issues in the ETL process and take corrective actions to ensure data integrity and accuracy.

#### Rationale
In the video streaming industry, reporting accuracy and reliability are paramount for stakeholders, including content partners and product teams. Any disruptions or discrepancies in the event sequences during the ETL process can lead to inaccuracies in reporting, affecting business decisions and partnerships. Therefore, it is essential to implement robust monitoring mechanisms to promptly detect and rectify out-of-sequence events.

#### Research Question
How can LSTM and GRU models monitor event sequences in the client event emitting process and detect out-of-sequence events?

#### Data Sources
The data used in this project consists of client event logs. The events enter the data pipeline and proceed through a TVM ETL. These logs contain information about events specific to the TVM calculation, timestamps, and other relevant attributes.

#### Methodology
The methodology involves the following steps:
1. **Exploring the Data:** Understanding the structure and characteristics of the event log data, identifying key features, and gaining insights into the underlying patterns.
2. **Data Preprocessing:** Cleaning and preparing the event log data for model training, including handling missing values, encoding categorical variables, and scaling numerical features.
3. **Model Development:** Building LSTM and GRU models to learn from the sequential nature of the event data and detect out-of-sequence events.
4. **Model Training:** Training the LSTM and GRU models on historical event sequences to learn patterns and relationships between events.
5. **Model Evaluation**: Assessing the performance of the LSTM and GRU models in detecting out-of-sequence events using appropriate evaluation metrics.

#### Results
Both the LSTM and GRU models were trained and evaluated on historical event sequences to detect out-of-sequence events in the TVM ETL process. The performance of both models was assessed using accuracy and loss metrics.

### LSTM Model:
- **Accuracy:** The LSTM model achieved an accuracy of 89.85% on the test dataset.
- **Loss:** The test loss for the LSTM model was 0.3348.

### GRU Model:
- **Accuracy:** Similarly, the GRU model achieved an accuracy of 89.55% on the test dataset.
- **Loss:** The test loss for the GRU model was 0.3484.

Both models demonstrated strong performance in detecting out-of-sequence events, with the LSTM model slightly outperforming the GRU model regarding accuracy and loss. The LSTM Model will be used to monitor the out-of-event sequences.
Further Model Validation: Conduct additional validation checks to ensure the LSTM model accurately adheres to the event sequencing rules. This may involve testing the models on diverse datasets and edge cases to confirm their reliability in effectively detecting out-of-sequence events.
Model Refinement: Fine-tune the LSTM model based on the insights gained from the validation process. Adjust model parameters, architecture, and training strategies to enhance performance, accuracy, and robustness in monitoring event sequences for the TVM ETL process.


#### Outline of Project
- [Exploratory Data Analysis Notebook](#) - Exploring the event log data and identifying patterns.
- [LSTM Model Notebook](#) - Building and training the LSTM model for event sequence monitoring.
- [GRU Model Notebook](#) - Developing and evaluating the GRU model for detecting out-of-sequence events.

##### Contact and Further Information
For inquiries or additional information, please contact krista.rime@pluto.tv
