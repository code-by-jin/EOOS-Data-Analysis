# EOOS-Data-Analysis
This is the repo for the paper "Sensor-based evaluation of a Urine Trap toilet in a shared bathroom". 

## Requirements
Python 3.7.4 <br>
skimage<br>
sklearn<br>
cv2<br>
matplotlib<br>

## Event Detection
- Organize readings (`.csv`/`.xlsx`) from weight scales and the flowmeter sensor into `\data\$period$\$date$\`
- Function `analyze_one_day` in `main_util.py` extract events based on the criteria described in the paper.  

## Event Classification
`main.ipynb` extracts events from all three periods and cluster events into two groups based on GMM.



