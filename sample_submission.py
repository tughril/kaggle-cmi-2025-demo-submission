import os

import polars as pl

import kaggle_evaluation.cmi_inference_server


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Replace this function with your inference code.
    # You can return either a Pandas or Polars dataframe, though Polars is recommended.
    # Each prediction (except the very first) must be returned within 30 minutes of the batch features being provided.
    print("called")
    print("Sequence shape:", sequence.shape)
    print("Sequence columns:", sequence.columns)
    print("Demographics shape:", demographics.shape)
    print("Demographics columns:", demographics.columns)
    print("Demographics data:")
    print(demographics)
    return "Text on phone"


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "./data/test.csv",
            "./data/test_demographics.csv",
        )
    )
