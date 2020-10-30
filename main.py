import os
import logging
from typing import Callable, List

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from starlette.requests import Request
from transformers import ZeroShotClassificationPipeline
from transformers import pipeline

TRF_MODEL_NAME = os.getenv("TRF_MODEL_NAME", "joeddav/xlm-roberta-large-xnli")

app = FastAPI(title="zero-shot-trf", openapi_url="/v1/openapi.json")
logger = logging.getLogger("zero-shot-trf")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level="INFO")


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Loading zero shot pipeline...")

        # models:
        #    - facebook/bart-large-mnli => default (english, from facebook)
        #    - joeddav/xlm-roberta-large-xnli => multilingual, by Joe from huggingface
        #    - valhalla/distilbart-mnli-12-6 => smaller model

        classifier: ZeroShotClassificationPipeline = pipeline(
            "zero-shot-classification", model=TRF_MODEL_NAME
        )
        app.classifier = classifier
        logger.info("Pipeline loaded.")

    #
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        app.classifier = None

    return shutdown


def get_classifier(request: Request) -> ZeroShotClassificationPipeline:
    return request.app.classifier


app.add_event_handler("startup", start_app_handler(app))
app.add_event_handler("shutdown", stop_app_handler(app))


class PredictRequest(BaseModel):
    """ """
    sequences: List[str]
    candidates: List[str]
    multi_class: bool = False
    hypothesis_template: str = Field("This example is {}.")


class PredictedLabel(BaseModel):
    """ """
    label: str
    score: float


class PredictSequenceResp(BaseModel):
    """ """
    sequence: str
    labels: List[PredictedLabel]


@app.get("/model-info")
async def model_info(
    classifier: ZeroShotClassificationPipeline = Depends(get_classifier),
):
    """ """
    return classifier.model.config.to_dict()


@app.post("/predict", response_model=List[PredictSequenceResp])
async def predict(request: PredictRequest, classifier: ZeroShotClassificationPipeline = Depends(get_classifier)):
    """ """
    predictions = classifier(
        request.sequences,
        request.candidates,
        multi_class=request.multi_class,
        hypothesis_template=request.hypothesis_template
    )
    #
    if len(request.sequences) == 1:
        predictions = [predictions]
    #
    resp = [
        PredictSequenceResp(
            sequence=pred["sequence"],
            labels=[
                PredictedLabel(
                    label=label,
                    score=score,
                )
                for label, score in zip(pred["labels"], pred["scores"])
            ]
        )
        for pred in predictions
    ]
    return resp


# def test():
#     clf: ZeroShotClassificationPipeline = pipeline("zero-shot-classification")
#     sequences = [
#         "Talent Acquisition Manager",
#         "Head of Recruitment",
#         "Client Success Director"
#     ]
#     candidates = [
#         "Sales",
#         "CEO",
#         "Founder",
#         "Human resources",
#         "Other",
#         "Growth",
#         "Customer success",
#     ]
#
#     hypothesis_template = "This example is {}."
#     hypothesis_template = "This job title is about {}."
#     hypothesis_template = "This example is {}."
#     hypothesis_template = "This job title belong to {}."
#
#     r = clf(
#         sequences, candidates, multi_class=False, hypothesis_template=hypothesis_template
#     )
#     pprint(r)
