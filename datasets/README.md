# Datasets for tensor time series
| Subject | Name      | Temporal Resolution | Statistics           | Description             | Type   |
|----------|-----------|---------------------|----------------------|--------------------------|--------|
| Traffic  | METR-LA   | 5 minutes            | (34,272, 207)         | traffic speed data       | MTS |
| Traffic  | PEMS-BAY  | 5 minutes            | (52,116, 326)         | Traffic speed and flow   | MTS |
| Traffic  | PEMS-2    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | LargeST   | 5 minutes            | (525,888, 8600)       | Traffic sensors          | MTS |
| Traffic  | TrafficBJ | 5 minutes            | (21,600, 3126)        | Traffic speeds           | MTS |
| Traffic  | TaxiBJ    | 30 minutes           | 4*(7220, 2, 32, 32)   | Crowd flows               | Tensor |
| Traffic  | JONAS-NYC | 30 minutes           | 2*(4800, 16, 8, 2)    | Demand & Supply          | Tensor |
| Traffic  | JONAS-DC  | 1 hour               | 2*(2400, 9, 12, 2)    | Demand & Supply          | Tensor |
| Traffic  | COVID-CHI | 2 hours              | 3*(6600, 14, 8, 2)    | Demand & Supply          | Tensor |
| Traffic  | COVID-US  | 1 hour               | (4800, 51, 10)        | Travel purpose            | Tensor |
| Traffic  | BikeNYC   |   -\| -     | trip records from June,2013 to January,2024            | Bike trip records         | Record |
| Traffic  | TaxiNYC   | -\| -           | 22,394,490 trip records           | Taxi trip records         | Record |
| Finance  | M4        | 1 hour ~ 1 year      | 100,000 time series   | Time series               | TS |
| Finance  | M5        | 1 day                | (30490, 1947), (30490, 1919) (train & valid) | Walmart sales forecast   | MTS |
| Finance  | NASDAQ 100   | 1 minute              | (391*191, 104), (390*191, 104) | Stock prices             | MTS |
| Finance  | Stock Market Dataset     | 1 day                | 8049*(D, 7)           | NASDAQ stock prices       | MTS |
| Finance  | CrypTop12    | 1 day                | (1255, 12, 7)& Tweets       | NASDAQ stock prices       | Tensor |
| Finance  | stocknet-dataset| 1 day                | (88, 731, 5)          | Stock price movement      | Tensor |
| Finance  | CSI300    | 1 day                |                       | Stock market index        | TS |
| Telecom  | Milan&Trentino   | 15 minutes            | (T, 100, 100, 5)      | 5 types of CDRs                      | Tensor |
| Health   | PTB-XL       | 2 ms, 10 ms          | Sample rate 100Hz, 500Hz for 10s | ECG dataset               | TS |
| Health   | MIMIC-III | various               |                       | Clinical data             | Series |
| Health   | MIT-BIH Arrhythmia Database | 2.78 ms               | 48 records, 2 input channels             | ECG dataset               | MTS |
| Health   | PTB Diagnostic ECG Database    | 1 ms                  | 549 records, 16 input channels           | ECG dataset               | MTS |
| Weather  | Shifts-Weather   |  -\| -   | (3129592, 129)        | Weather prediction        | MTS |
| Energy   | ElectricityLoad| 15 minutes           | (140256,370)         | Electricity consumption   | MTS |
