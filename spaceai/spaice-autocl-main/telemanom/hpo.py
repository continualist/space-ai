import shutil
import os
import tempfile

import torch
from telemanom.modeling import Model
import telemanom.helpers as helpers
import torch.optim as optim

import ray
from ray import tune, train
from ray.air import session
from ray.air import RunConfig
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch
import optuna
import operator
import logging

logger = logging.getLogger('telemanom')

class HPO():
    """
        class to manage the HPO (classical or adaptive) stage 1
    """

    def __init__(self, config, run_id):
        self.config = config
        self. run_id = run_id
        self.channel = None

        if self.config.hpo_type == "classical":
            self.sampler = optuna.samplers.RandomSampler()
        elif self.config.hpo_type == "adaptive":
            self.sampler = optuna.samplers.TPESampler()
            self.hps_importance = None
        else:
            pass # TODO: manage
        self.optuna_search = None

        if self.config.model_architecture == "ESN":
            # (edited search space)
            self.search_space = {
                "hidden_size_1": tune.randint(150, 1000),
                "hidden_size_2": tune.randint(150, 1000),
                "input_scaling": tune.uniform(0.1, 0.999),
                "leakage": tune.uniform(0.5, 1.1), # integration constant, controls the loss of information in the nodes
                "rho": tune.uniform(0.1, 0.99), # spectral radius
                "l2": tune.loguniform(1e-10, 1e-4) # l2 readout regularization
            }
        else: # LSTM
            self.search_space = {
                "lr":  tune.loguniform(1e-4, 1e-2),
                "weight_decay": tune.loguniform(1e-8, 1e-3),
                "dropout": tune.uniform(0.1, 0.5),
                "hidden_size_1": tune.randint(20, 100),
                "hidden_size_2": tune.randint(20, 100)
            }

    def execute_hpo(self, channel, num_samples=5, num_samples2=3, resources={"cpu": 2}):
        self.channel = channel
        self.optuna_search = OptunaSearch(metric="MSE", mode="min", sampler=self.sampler)
        tuner = tune.Tuner(
            tune.with_parameters(
                tune.with_resources(train_ad, resources),
                cfg = self.config,
                id=self.run_id,
                channel=self.channel
            ),
            tune_config=tune.TuneConfig(
                search_alg=self.optuna_search,
                num_samples=num_samples,
                # scheduler=self.scheduler,
            ),
            param_space=self.search_space, #
            run_config=RunConfig(
                local_dir=os.path.abspath("./hpo_results"),
                log_to_file=False,
            )
        )

        results = tuner.fit()
        #logger.info('End of HPO1')

        if self.config.hpo_type == 'adaptive':
            # Perform pruning of hyperparameters and values
            pruned_search_space = self.prune_search_space(results)

            # Execute additional HPO using the updated search space
            self.optuna_search = OptunaSearch(metric="MSE", mode="min", sampler=self.sampler)
            tuner = tune.Tuner(
                tune.with_parameters(
                    tune.with_resources(train_ad, {'cpu': self.config.cpu_adaptive}),
                    cfg=self.config,
                    id=self.run_id,
                    channel=self.channel
                ),
                tune_config=tune.TuneConfig(
                    search_alg=self.optuna_search,
                    num_samples=num_samples2,
                ),
                param_space=pruned_search_space,
                run_config=RunConfig(
                    local_dir=os.path.abspath("./hpo_results"),
                    log_to_file=False,
                )
            )

            results = tuner.fit()
            # logger.info('End of HPO2')
        else:
            pass # TODO: manage

        # Get best trial
        best_result = results.get_best_result(metric="MSE", mode="min")
        best_metric = best_result.metrics
        best_hps = best_result.config

        # Remove folder for efficient-memory run at the end of each channel
        shutil.rmtree('./hpo_results')

        return best_metric, best_hps

    def prune_search_space(self, results):
        # Perform pruning using techniques like fANOVA based on feature importance
        self.hps_importance = self.get_hps_importance()
        top_hps = self.top_n_items(self.hps_importance, k=2)  # Select top 2 most important hyperparameters

        # Update the search space with the top hyperparameters and their values from HPO1
        pruned_search_space = {}
        alpha, beta = 0.75, 1.25
        for hp_name, hp_value in self.search_space.items():
            if hp_name in top_hps.keys():
                # Narrow the search space around the best value found in HPO1 for important hyperparameters
                best_value_hpo1 = results.get_best_result(metric="MSE", mode="min").config[hp_name]
                if isinstance(hp_value, tune.search.sample.Integer):
                    pruned_search_space[hp_name] = tune.randint(int(best_value_hpo1 * alpha), int(best_value_hpo1 * beta))
                else:  # Uniform/loguniform hyperparameter
                    pruned_search_space[hp_name] = tune.uniform(best_value_hpo1 * alpha, best_value_hpo1 * beta)
            else:
                # Keep the best value found in HPO1 for non-important hyperparameters
                best_value_hpo1 = results.get_best_result(metric="MSE", mode="min").config[hp_name]
                pruned_search_space[hp_name] = best_value_hpo1

        return pruned_search_space

    def get_hps_importance(self):
        # Importance HPs
        importance = optuna.importance.get_param_importances(self.optuna_search._ot_study, evaluator=None, params=None,
                                                             target=None, normalize=True)
        # logger info hps importance
        logger.info(" ====  Importance  ====")
        for key, value in importance.items():
            logger.info("    {}: {}".format(key, value))
        logger.info(" ====  -  ====")

        return importance

    def top_n_items(self, importance_dict, k=2):
        sorted_items = sorted(importance_dict.items(), key=operator.itemgetter(1), reverse=True)

        return dict(sorted_items[:k])


def train_ad(config, cfg, id, channel):
    cfg.layers = [config['hidden_size_1'], config['hidden_size_2']]
    if cfg.model_architecture == "ESN":
        cfg.leakage = config['leakage']
        cfg.input_scaling = config['input_scaling']
        cfg.rho = config['rho']
        cfg.l2 = config['l2']
        #
        model = Model(cfg, id, channel)
        MSE = model.train_new(channel)
        train.report(MSE)
    else:  # LSTM
        cfg.learning_rate = config['lr']
        cfg.weight_decay = config['weight_decay']
        cfg.dropout = config['dropout']
        #
        model = Model(cfg, id, channel)
        model.train_new(channel)
        train.report({"MSE": model.valid_loss})
