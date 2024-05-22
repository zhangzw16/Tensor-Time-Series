# Datasets for tensor time series
Stay tuned for more datasets for tensor time series.
| Subject | Name      | Temporal Resolution | Statistics           | Description             | Type   |
|----------|-----------|---------------------|----------------------|--------------------------|--------|
| Traffic  | [METR-LA](https://github.com/liyaguang/DCRNN)   | 5 minutes            | (34272, 207)         | traffic speed data       | MTS |
| Traffic  | [PEMS-BAY](https://zenodo.org/records/5724362)  | 5 minutes            | (52116, 326)         | Traffic speed and flow   | MTS |
| Traffic  | [PEMS-03](https://drive.google.com/drive/folders/17R3RiKrDaV4nsXb-ADY7l51RRPTUx8vr)    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | [PEMS-04](https://drive.google.com/drive/folders/17R3RiKrDaV4nsXb-ADY7l51RRPTUx8vr)    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | [PEMS-07](https://drive.google.com/drive/folders/17R3RiKrDaV4nsXb-ADY7l51RRPTUx8vr)    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | [PEMS-08](https://drive.google.com/drive/folders/17R3RiKrDaV4nsXb-ADY7l51RRPTUx8vr)    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | [PEMS-20](https://drive.google.com/drive/folders/17R3RiKrDaV4nsXb-ADY7l51RRPTUx8vr)    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | [PEMS-20](https://drive.google.com/drive/folders/17R3RiKrDaV4nsXb-ADY7l51RRPTUx8vr)    | 5 minutes            | (61,056, 325)         | Traffic speeds           | MTS |
| Traffic  | [LargeST](https://github.com/liuxu77/LargeST)   | 5 minutes            | (525888, 8600)       | Traffic sensors          | MTS |
| Traffic  | [TrafficBJ](https://github.com/deepkashiwa20/Urban_Concept_Drift) | 5 minutes            | (21600, 3126)        | Traffic speeds           | MTS |
| Transport  | [TaxiBJ](https://github.com/TolicWang/DeepST/tree/master/data/TaxiBJ)    | 30 minutes           | 4*(7220, 2, 32, 32)   | Crowd flows               | Tensor |
| Transport  | [JONAS-NYC](https://github.com/underdoc-wang/EAST-Net/tree/main) | 30 minutes           | 2*(4800, 16, 8, 2)    | Demand & Supply          | Tensor |
| Transport  | [JONAS-DC](https://github.com/underdoc-wang/EAST-Net/tree/main)  | 1 hour               | 2*(2400, 9, 12, 2)    | Demand & Supply          | Tensor |
| Transport  | [COVID-CHI](https://github.com/underdoc-wang/EAST-Net/tree/main) | 2 hours              | 3*(6600, 14, 8, 2)    | Demand & Supply          | Tensor |
| Travel  | [COVID-US](https://github.com/underdoc-wang/EAST-Net/tree/main)  | 1 hour               | (4800, 51, 10)        | Travel purpose            | Tensor |
| Transport  | [BikeNYC](https://citibikenyc.com/system-data)   |   -\| -     | trip records from June,2013 to January,2024            | Bike trip records         | Record |
| Transport  | [TaxiNYC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)   | -\| -           | 22394,490 trip records           | Taxi trip records         | Record |
| Finance  | [M4](https://github.com/Mcompetitions/M4-methods)        | 1 hour ~ 1 year      | 100,000 time series   | Time series               | TS |
| Finance  | [M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)        | 1 day                | (30490, 1947), (30490, 1919) (train & valid) | Walmart sales forecast   | MTS |
| Finance  | [NASDAQ 100](https://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)   | 1 minute              | (391*191, 104), (390*191, 104) | Stock prices             | MTS |
| Finance  | [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/data)     | 1 day                | 8049*(D, 7)           | NASDAQ stock prices       | MTS |
| Finance  | [CrypTop12](https://github.com/am15h/CrypTop12/tree/main/tweet/preprocessed/eth)    | 1 day                | (1255, 12, 7)& Tweets       | NASDAQ stock prices       | Tensor |
| Finance  | [Stocknet-Dataset](https://github.com/yumoxu/stocknet-dataset/tree/master)| 1 day                | (731, 88, 5)          | Two-year price movements of 88 stocks     | Tensor |
| Finance  | [CSI300](https://www.marketwatch.com/investing/index/000300/download-data?countrycode=xx&mod=mw_quote_tab)    | 1 day                |                       | Stock market index        | TS |
| Telecom  | [Milan&Trentino](https://www.nature.com/articles/sdata201555)   | 15 minutes            | (T, 100, 100, 5)      | 5 types of CDRs                      | Tensor |
| Health   | [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)       | 2 ms, 10 ms          | Sample rate 100Hz, 500Hz for 10s | ECG dataset               | TS |
| Health   | [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) | various               |                       | Clinical data             | Series |
| Health   | [MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/) | 2.78 ms               | 48 records, 2 input channels             | ECG dataset               | MTS |
| Health   | [PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/)    | 1 ms                  | 549 records, 16 input channels           | ECG dataset               | MTS |
| Weather  | [Shifts-Weather](https://github.com/Shifts-Project/shifts/tree/main/weather)   |  -\| -   | (3129592, 129)        | Weather prediction        | MTS |
| Energy   | [ElectricityLoad](http://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)| 15 minutes           | (140256,370)         | Electricity consumption   | MTS |
