from datetime import datetime
import json
import numpy as np

from scaling_experiments import run_experiment

DEFAULT_PARAMS = dict(
    delta_type="none",  # 'none', 'shared', 'multi',
    initial_sd=1.0e-04,
    init_method="xavier",
    activation="tanh",  # 'relu', 'tanh', 'linear'
    dim=10,
    dataset="ODE",  # 'ODE', 'mnist'
    optimizer_name="adam",
    num_epochs=200,
    epsilon=1.0e-03,
    train_size=1024,
    test_size=256,
    batch_size=50,
    lr=1.0e-03,
    path="./scaling/",
    save=True,
    min_depth=3,
    max_depth=1000,
    base=1.2,  # base**n < max_depth
)

SCALING_PARAMS = [
    dict(
        DEFAULT_PARAMS,
        **dict(
            path=DEFAULT_PARAMS["path"] + "dataset-mnist/act-tanh/delta-shared/",
            dataset="ODE",
            num_epochs=10,
            dim=10,
            batch_size=32,
            lr=1e-2,
            epsilon=1e-3,
            activation="tanh",
            delta_type="shared",  # 'multi' | 'shared'
            initial_sd=0.01,
            init_method="xavier-depth",
            min_depth=3,
            max_depth=10,
            base=1.25992105,
        ),
    )
]


def run():
    NOW = datetime.now().strftime("%Y-%m-%d-%H-%M") + "/"
    for i, params_dict in enumerate(SCALING_PARAMS):
        params_dict["path"] += NOW
        print(f"Scaling experiments, starting {i + 1}/{len(SCALING_PARAMS)}.")
        print("Path: ", params_dict["path"])
        run_experiment(**params_dict)

        with open(params_dict["path"] + "params_dict.json", "w") as fp:
            json.dump(params_dict, fp)


if __name__ == "__main__":
    run()
