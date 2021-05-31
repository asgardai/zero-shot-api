import logging
import os
import tempfile
from typing import Callable, List

import numpy as np
from fastapi import FastAPI, Depends
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from pydantic import BaseModel, Field
from starlette.requests import Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model: AutoModelForSequenceClassification = (
    AutoModelForSequenceClassification.from_pretrained("bert")
)

TRF_MODEL_PATH = os.getenv("TRF_MODEL_PATH")
TRF_MODEL_NAME = os.getenv("TRF_MODEL_NAME", "joeddav/xlm-roberta-large-xnli")

app = FastAPI(title="zero-shot-trf", openapi_url="/v1/openapi.json")
logger = logging.getLogger("zero-shot-trf")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level="INFO")


def load_onnx_model(model_path) -> InferenceSession:
    """ """
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    #
    return session


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Loading zero shot pipeline...")

        # models:
        #    - facebook/bart-large-mnli => default (english, from facebook)
        #    - joeddav/xlm-roberta-large-xnli => multilingual, by Joe from huggingface
        #    - valhalla/distilbart-mnli-12-6 => smaller model
        # logger.info(f"Loading zero-shot pipeline with model: {TRF_MODEL_NAME}")
        # classifier: ZeroShotClassificationPipeline = pipeline(
        #     "zero-shot-classification", model=TRF_MODEL_NAME
        # )

        if TRF_MODEL_PATH.startswith("gs://"):
            logger.info("Dowloading model from ")
            local_model_path = download_from_gcs(TRF_MODEL_PATH)
        else:
            local_model_path = TRF_MODEL_PATH
        #
        if local_model_path.endswith(".zip"):
            logger.info("Unzipping model")
            local_model_path = unzip_and_get_onnx_file_path(local_model_path)
        #
        logger.info(f"Loading onnx session with model: {TRF_MODEL_PATH} ...")
        onnx_session = load_onnx_model(local_model_path)
        logger.info(f"Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(TRF_MODEL_NAME)

        inputs = tokenizer.encode_plus("desfes  efsd")
        res = model.forward(inputs)
        #
        app.onnx_session = onnx_session
        app.tokenizer = tokenizer

        logger.info("Pipeline loaded.")

    #
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        app.onnx_session = None
        app.tokenizer = None
        # app.classifier = None

    return shutdown


def convert_to_onnx_inputs(inputs, model_signature):
    """ """
    input_names = [x.name for x in model_signature]
    return {k: v for k, v in inputs.items() if k in input_names}


def compute_zero_shot(outputs, sequences, candidates, multi_class: bool):
    """
    Compute zero shot classification between sequences and candidates using prediction
    from a pre-trained model on mnli/xnli.

    Taken from huggingface zero-shop-classification pipeline

    :param outputs:
    :param sequences:
    :param candidates:
    :param multi_class:
    :return:
    """
    num_sequences = len(sequences)
    reshaped_outputs = outputs.reshape(len(sequences), len(candidates), -1)

    if len(candidates) == 1:
        multi_class = True

    if not multi_class:
        # softmax the "entailment" logits over all candidate labels
        entail_logits = reshaped_outputs[..., -1]
        scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
    else:
        # softmax over the entailment vs. contradiction dim for each label independently
        entail_contr_logits = reshaped_outputs[..., [0, -1]]
        scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(
            -1, keepdims=True
        )
        scores = scores[..., 1]

    results = []
    for i in range(num_sequences):
        top_inds = list(reversed(scores[i].argsort()))
        results.append(
            {
                "sequence": sequences[i],
                "labels": [candidates[i] for i in top_inds],
                "scores": scores[i][top_inds].tolist(),
            }
        )
    #
    return results


def get_model_session(request: Request) -> InferenceSession:
    return request.app.onnx_session


def get_tokenizer(request: Request):
    return request.app.tokenizer


app.add_event_handler("startup", start_app_handler(app))
app.add_event_handler("shutdown", stop_app_handler(app))


class PredictRequest(BaseModel):
    """ """

    sequences: List[str]
    candidates: List[str]
    multi_class: bool = False
    hypothesis_template: str = Field("This example is {}.")
    batch_size: int = 10


class PredictedLabel(BaseModel):
    """ """

    label: str
    score: float


class PredictSequenceResp(BaseModel):
    """ """

    sequence: str
    labels: List[PredictedLabel]


@app.get("/model-info")
async def model_info():
    """ """
    return {"model_path": TRF_MODEL_PATH, "model_name": TRF_MODEL_NAME}


@app.get("/")
async def default():
    """ """
    return {"zero": "shot"}


@app.post("/predict", response_model=List[PredictSequenceResp])
def predict(
    request: PredictRequest,
    model_session: InferenceSession = Depends(get_model_session),
    tokenizer=Depends(get_tokenizer),
):
    """ """
    predictions = []
    for batch in partition_all(request.sequences, n=request.batch_size):
        #
        sequence_pairs = []
        for sequence in batch:
            sequence_pairs.extend(
                [
                    [sequence, request.hypothesis_template.format(label)]
                    for label in request.candidates
                ]
            )

        inputs = tokenizer(
            sequence_pairs,
            add_special_tokens=True,
            return_tensors=None,
            padding=True,
            truncation="only_first",
        )
        onnx_inputs = convert_to_onnx_inputs(
            inputs, model_signature=model_session.get_inputs()
        )
        outputs = model_session.run(None, onnx_inputs)[0]
        batch_predictions = compute_zero_shot(
            outputs, batch, request.candidates, request.multi_class
        )
        #
        predictions.extend(batch_predictions)

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
            ],
        )
        for pred in predictions
    ]
    return resp


def partition_all(iterable, n=1):
    """ Iterate over an iterable in batches """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def download_from_gcs(gcs_url: str, dest_dir: str = None) -> str:
    """ """
    from google.cloud import storage

    #
    if dest_dir is None:
        dest_dir = tempfile.gettempdir()

    gcs_client = storage.Client()

    tmp = gcs_url.split("gs://")[1].split("/")
    bucket = tmp[0]
    key = "/".join(tmp[1:]) if len(bucket) > 1 else "/"
    #
    bucket = gcs_client.get_bucket(bucket)
    blob = bucket.blob(key)
    local_fname = os.path.join(dest_dir, tmp[-1])
    blob.download_to_filename(local_fname)
    #
    return local_fname


def unzip_and_get_onnx_file_path(zip_path):
    """ """
    from zipfile import ZipFile

    zf = ZipFile(zip_path)
    zf.extractall(os.path.dirname(zip_path))
    extracted_path = os.path.join(
        os.path.dirname(zip_path), os.path.basename(zip_path)[:-4]
    )
    #
    onnx_files = [x for x in os.listdir(extracted_path) if x.endswith(".onnx")]
    if len(onnx_files) == 1:
        return os.path.join(extracted_path, onnx_files[0])
    else:
        raise ValueError("Found no or several onnx files in model archives")
