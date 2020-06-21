from transformers.data.processors.glue import (
    ColaProcessor,
    MnliProcessor,
    MrpcProcessor,
    QnliProcessor,
    QqpProcessor,
    RteProcessor,
    Sst2Processor,
    StsbProcessor,
    WnliProcessor,
)
from hans.utils_hans import HansProcessor

processor_dict = {
    "mrpc": MrpcProcessor,
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliProcessor,
    "sst-2": Sst2Processor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "sts-b": StsbProcessor,
    "hans": HansProcessor,
}


def get_dataset_dict(data_args):
    return {
        "mrpc": data_args.data_dir + "/MRPC",
        "cola": data_args.data_dir + "/CoLA",
        "mnli": data_args.data_dir + "/MNLI",
        "mnli-mm": data_args.data_dir + "/MNLI",
        "sst-2": data_args.data_dir + "/SST-2",
        "rte": data_args.data_dir + "/RTE",
        "wnli": data_args.data_dir + "/WNLI",
        "qqp": data_args.data_dir + "/QQP",
        "qnli": data_args.data_dir + "/QNLI",
        "sts-b": data_args.data_dir + "/STS-B",
        "hans": data_args.data_dir + "/hans",
    }
