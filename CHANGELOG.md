## 0.2.0 (2024-12-03)

### Feat

- Add callback handling on benchmarks
- Adaptation of esa dataset and benchmark
- Add monitor of disk usage for each channel in nasa experiments
- Add multi-input features in nasa dataset
- Add callback handling on benchmarks
- Adaptation of esa dataset and benchmark
- Add monitor of disk usage for each channel in nasa experiments
- Add multi-input features in nasa dataset
- implementation of esa benchmark and experiments
- Restructure codebase and include Telemanom and NASA benchmark
- Refactor experiment management
- Reimplement ESAAD to load single channels
- Incluse ESAAD benchmark
- Include NASA benchmark and telemanom
- add type annotations to variables

### Fix

- Change makefile targets
- Change url of nasa dataset and the way to extract and process the tar file
- ESA benchmark aligned with Nasa
- implement reduce_out in LSTM
- Telemanom fix
- Add flush_detector and remove debug functions
- Telemanom fix
- ESN add clip on prediction and chenge default values
- Compute precision and recall defaults
- Fix dependency torch-rc installing it with pip
- Fix readme torch-rc dependency
- Fix dependencies
- Comment self._predictor.reset_state in telemanom
- Compute volatility with respect the first dimension
- Check anomaly_seq_df length instead of anomaly_df
- Change log string formatting
- Use device during prediction
- align ESN and LSTM prediction output tensor for computing the mean
- ESA benchmark aligned with Nasa
- implement reduce_out in LSTM
- Telemanom fix
- Add flush_detector and remove debug functions
- Telemanom fix
- ESN add clip on prediction and chenge default values
- Compute precision and recall defaults
- Fix dependency torch-rc installing it with pip
- Fix readme torch-rc dependency
- Fix dependencies
- Comment self._predictor.reset_state in telemanom
- Compute volatility with respect the first dimension
- Check anomaly_seq_df length instead of anomaly_df
- Fix telemanom bugs on compare_to_epsilon and prune_anomalies, fix metrics, and fix anomalies on dataset. Add rss memory monitor
- Fix telemanom algorithm (in details the if condition of compare epsilon, the indices of assignemnt of scores and EWMA algorithm). Fix LSTM stateful
- Save model on early stopping
- Change buffering rationale in nasa and esa benchmarks
- Handle when nasa channel has not abonalies and add requests library to poetry
- add missing arguments to evaluate_sequences() function
- resolve bugs caused by type annotations

### Refactor

- Scatter utils across packages
- efficient loading getitem nasa dataset
- Refector  getitem method of nasa detaset
- Centralize reduce_out option in seq_model
- efficient loading getitem nasa dataset
- Refector  getitem method of nasa detaset
- dataset esa changing the save and loading with adaptation of the flag of resampling to work also with the saved preprocess. Usage of train and test split during resampling
- refactoring telecommand ancoding method of esa dataset loader
- Adapt esa dataset to the code structure on the basis of nasa
- Adapt Telemanom to new codebase structure
- move retrain function from Detector class to Trainer class
- move evaluate_sequences() and log_final_stats() to helpers.py, Config class to config.py.
- remove self.train_new(channel) business logic from initialization
- rename variables to snake_case to fix pylint invalid-name warnings
- **spaice-autocl**: remove autocl part to avoid patent conflicts
