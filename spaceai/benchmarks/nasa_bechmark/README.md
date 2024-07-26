# Codebase for NASA Benchmark

Clone the repo:

```sh
git clone https://github.com/continualist/space-ai.git
```


#### To run with local or virtual environment

From ``` ./spaceai/benchmarks/nasa_benchmark ```, curl and unzip data:

```sh
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
```

Install dependencies using **python 3.6+** (recommend using a virtualenv or conda):

```sh
pip install -r requirements.txt
```

Begin processing (from ``` ./spaceai/benchmarks/nasa_benchmark ```):

```sh
# rerun experiment labeled anomlies
python run_exps.py -l labeled_anomalies.csv

# run without labeled anomalies
python run_exps.py
```

# NASA Data (SMAP, MSL)

## Using your own data

Pre-split training and test sets must be placed in directories named `data/train/` and `data/test`. One `.npy` file should be generated for each channel or stream (for both train and test) with shape (`n_timesteps`, `n_inputs`). The filename should be a unique channel name or ID. The telemetry values being predicted in the test data *must* be the first feature in the input.

For example, a channel `T-1` should have train/test sets named `T-1.npy` with shapes akin to `(4900,61)` and `(3925, 61)`, where the number of input dimensions are matching (`61`). The actual telemetry values should be along the first dimension `(4900,1)` and `(3925,1)`.

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

# Processing

Each time the system is started a unique datetime ID (ex. `2018-05-17_16.28.00`) will be used to create the following
- a **results** file (in `results/`) that extends `labeled_anomalies.csv` to include identified anomalous sequences and related info
- a **data subdirectory** containing data files for created models, predictions, and smoothed errors for each channel. A file called `params.log` is also created that contains parameter settings and logging output during processing.
