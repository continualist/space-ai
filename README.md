# SpaceAI
Repository for providing off-the-shelf benchmarks for AI-based aerospace applications.


### Getting started

**Installing this codebase requires Python 3.10 or 3.11.**
Run the following commands within your python virtual environment:

```sh
pip install poetry

git clone https://github.com/vdecaro/torch-rc.git
cd torch-rc
poetry install
cd ..

git clone https://github.com/continualist/space-ai.git
cd space-ai
poetry install
```

# NASA Data (SMAP, MSL)


## Anomaly labels and metadata

The anomaly labels and metadata are available in `labeled_anomalies.csv`, which includes:

- `channel id`: anonymized channel id - first letter represents nature of channel (P = power, R = radiation, etc.)
- `spacecraft`: spacecraft that generated telemetry stream
- `anomaly_sequences`: start and end indices of true anomalies in stream
- `class`: the class of anomaly (see paper for discussion)
- `num values`: number of telemetry values in each stream

To provide your own labels, use the `labeled_anomalies.csv` file as a template. The only required fields/columns are `channel_id` and `anomaly_sequences`. `anomaly_sequences` is a list of lists that contain start and end indices of anomalous regions in the test dataset for a channel.

## Dataset and performance statistics:

#### Data
|								  | SMAP 	  | MSL		 | Total   |
| ------------------------------- |	:-------: |	:------: | :------:|
| Total anomaly sequences 		  | 69        | 36		 | 105	   |
| *Point* anomalies (% tot.)	  | 43 (62%)  | 19 (53%) | 62 (59%)|
| *Contextual* anomalies (% tot.) | 26 (38%)  | 17 (47%) | 43 (41%)|
| Unique telemetry channels		  | 55        | 27		 | 82	   |
| Unique ISAs					  | 28		  | 19		 | 47	   |
| Telemetry values evaluated	  | 429,735	  | 66,709   | 496,444 |

#### Performance (with default params specified in paper)
| Spacecraft		| Precision | Recall   | F_0.5 Score |
| ----------------- | :-------: | :------: | :------: |
| SMAP 		  		| 85.5%     | 85.5%	   | 0.71	  |
| Curiosity (MSL)	| 92.6%  	| 69.4%    | 0.69     |
| Total 			| 87.5% 	| 80.0%	   | 0.71     |

