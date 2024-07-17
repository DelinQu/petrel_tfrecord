import threading
import queue
import time
import os
from petrel_client.client import Client
from typing import Iterable


class MultiThreadedOSSIterator:
    def __init__(self, files_list, max_queue_size, num_producers):
        self.files_list = files_list
        self.max_queue_size = max_queue_size
        self.num_producers = num_producers
        self.share_data = queue.Queue(maxsize=max_queue_size)
        self.producer_threads = []
        self.consumer_thread = None

    def producer(self, index):
        for i in range(index, len(self.files_list), self.num_producers):
            file = self.files_list[i]
            print(f"Producer {index} producing: {file}")
            self.share_data.put(file)
            print(f"Producer {index} put {file} into queue")
            time.sleep(1)

    def consumer(self):
        while True:
            file = self.share_data.get()
            if file is None:
                break
            print(f"Consumer consuming: {file}")
            # 在这里添加处理文件的逻辑
            time.sleep(2)
            self.share_data.task_done()
        print("Consumer finished processing all files")

    def start_producers(self):
        for i in range(self.num_producers):
            t = threading.Thread(target=self.producer, args=(i,))
            t.start()
            self.producer_threads.append(t)

    def start_consumer(self):
        self.consumer_thread = threading.Thread(target=self.consumer)
        self.consumer_thread.start()

    def join_producers(self):
        for t in self.producer_threads:
            t.join()

    def join_consumer(self):
        self.share_data.put(None)  # 通知消费者线程已经没有更多的文件了
        self.consumer_thread.join()

    def process_files(self):
        self.start_producers()
        self.start_consumer()
        self.join_producers()
        self.join_consumer()
        print("All files have been processed.")


if __name__ == "__main__":
    files_list = ["file1", "file2", "file3", "file4", "file5", "file6", "file7", "file8"]
    max_queue_size = 3
    num_producers = 2

    processor = MultiThreadedOSSIterator(files_list, max_queue_size, num_producers)
    processor.process_files()


class DatasetBuilder(IterDataPipe[TFRecordExample]):
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
        self.file_path_list = self.file_client.get_file_iterator(os.path.join(self.base_path, self.name))
        self.file_path_list = sorted(filter(lambda x: f"{self.split}.tfrecord" in x, self.file_path_list))

        self.max_queue_size = max_queue_size
        self.num_producers = num_producers
        self.share_data = queue.Queue(maxsize=max_queue_size)
        self.producer_threads = []
        self.consumer_thread = None

    def producer(self, index):
        idxs = range(index, len(self.file_path_list), self.num_producers)
        datapipe = FileOpener(map(lambda x: self.files_list[x], idxs), mode="b")

    # def producer(self, index):
    #     for i in range(index, len(self.files_list), self.num_producers):
    #         file = self.files_list[i]
    #         print(f"Producer {index} producing: {file}")
    #         self.share_data.put(file)
    #         print(f"Producer {index} put {file} into queue")
    #         time.sleep(1)

    def consumer(self):
        while True:
            file = self.share_data.get()
            if file is None:
                break
            print(f"Consumer consuming: {file}")
            # 在这里添加处理文件的逻辑
            time.sleep(2)
            self.share_data.task_done()
        print("Consumer finished processing all files")

    def start_producers(self):
        for i in range(self.num_producers):
            t = threading.Thread(target=self.producer, args=(i,))
            t.start()
            self.producer_threads.append(t)

    def start_consumer(self):
        self.consumer_thread = threading.Thread(target=self.consumer)
        self.consumer_thread.start()

    def join_producers(self):
        for t in self.producer_threads:
            t.join()

    def join_consumer(self):
        self.share_data.put(None)  # 通知消费者线程已经没有更多的文件了
        self.consumer_thread.join()

    def process_files(self):
        self.start_producers()
        self.start_consumer()
        self.join_producers()
        self.join_consumer()
        print("All files have been processed.")

    def build():
        self.process_files()

    def __iter__(self):
        path = os.path.join(self.base_path, self.name)
        for oss_path, dict in self.file_client.get_file_iterator(path):
            if not f"{self.split}.tfrecord" in oss_path:
                continue
            yield self.file_client.get("s3://" + oss_path)

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
