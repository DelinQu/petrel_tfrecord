import queue
import warnings
from typing import Iterator
import threading
import queue
import time
import os
from petrel_client.client import Client

from .torch_data import (
    FileOpenerIterDataPipe,
    validate_pathname_binary_tuple,
    iterate_tfrecord_file,
    parse_tfrecord_sequence_example,
    TFRecordExample,
    IterDataPipe,
)

from .protobuf_template import _tfrecord_example_pb2 as example_pb2


class TFRecordDatasetIter(IterDataPipe[TFRecordExample]):
    def __init__(
        self,
        base_path="s3://songhaoming/open_x_embodiment_origin",
        split="train",
        name="roboturk/0.1.0",
        conf_path="~/vla-oss.conf",
        max_queue_size=1,
        num_producers=1,
    ) -> None:
        super().__init__()
        self.base_path = base_path
        self.split = "" if split == "all" else split
        self.name = name

        self.file_client = Client(conf_path)
        self.file_path_list = self.file_client.get_file_iterator(
            os.path.join(self.base_path, self.name)
        )
        self.file_path_list = sorted(
            filter(lambda x: f"{self.split}.tfrecord" in x, self.file_path_list)
        )

        self.max_queue_size = max_queue_size
        self.num_producers = num_producers
        self.share_data = queue.Queue(maxsize=max_queue_size)
        self.producer_threads = []
        self.consumer_thread = None

    def producer(self, index):
        idxs = range(index, len(self.file_path_list), self.num_producers)
        datapipe = FileOpenerIterDataPipe(
            map(lambda x: self.files_list[x], idxs), mode="b"
        )
        for data in datapipe:
            print(f"* Producer {index} producing: {data}")
            self.share_data.put(data)
            print(f"- Producer {index} put {data} into queue")
            time.sleep(1)
        self.share_data.put(None)
        print(f"Producer {index} finished producing all files and put a None")

    def start_producers(self):
        for i in range(self.num_producers):
            t = threading.Thread(target=self.producer, args=(i,))
            t.start()
            self.producer_threads.append(t)

    def join_producers(self):
        for t in self.producer_threads:
            t.join()

    def __iter__(self) -> Iterator[TFRecordExample]:
        self.finish_signals = 0
        self.start_producers()

        while True:
            data = self.share_data.get()
            if data is None:
                self.finish_signals += 1
                if self.finish_signals >= self.num_producers:
                    break
                continue
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                for example_bytes in iterate_tfrecord_file(data_stream):
                    example = example_pb2.SequenceExample()  # type: ignore
                    example.ParseFromString(example_bytes)  # type: ignore
                    yield parse_tfrecord_sequence_example(example, self.spec)
            except RuntimeError as e:
                warnings.warn(
                    f"Unable to read from corrupted tfrecord stream {pathname} due to: {e}, abort!"
                )
                raise e
            self.share_data.task_done()

        self.join_producers()
        print("All files have been processed.")

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
