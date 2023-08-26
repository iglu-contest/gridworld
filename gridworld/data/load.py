import os
import logging
import tarfile
import requests
from tqdm import tqdm

PUBLIC_SAS_TOKEN = "sp=rl&st=2021-11-01T16:22:02Z&se=2022-02-03T01:22:02Z&spr=https&sv=2020-08-04&sr=c&sig=Nz2bOp8rRWEwq1E7Ycg5B3VeTBWld1%2FLVAgrhtrDo%2Fs%3D"

def download(url, destination, data_prefix, description='downloading dataset into'):
    os.makedirs(data_prefix, exist_ok=True)
    r = requests.get(url, stream=True)
    CHUNK_SIZE = 1048576
    total_length = int(r.headers.get('content-length'))
    print(f'{description} into {data_prefix}')
    with open(destination, "wb") as f:
        with tqdm(desc=description, 
                  total=(total_length // CHUNK_SIZE) + 1) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(1)

if 'IGLU_SAS_TOKEN' not in os.environ:
    os.environ['IGLU_SAS_TOKEN'] = PUBLIC_SAS_TOKEN

logger = logging.getLogger(__name__)

class BlobFileDownloader:
    def __init__(self, local_blob_path=None):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError:
            raise ImportError('please install azure blob storage via pip: pip install azure-storage-blob')
        self.sas_token = os.getenv("IGLU_SAS_TOKEN")
        self.sas_url = "https://igludatacollection.blob.core.windows.net/iglu-data-task-2?" + self.sas_token
        self.container_client = ContainerClient.from_container_url(self.sas_url)
        self.local_blob_path = local_blob_path

    def list_blobs(self):
        blob_list = self.container_client.list_blobs()
        for blob in blob_list:
            print(blob.name + '\n')

    def __save_blob__(self,file_name,file_content):
        # Get full path to the file
        download_file_path = os.path.join(self.local_blob_path, file_name)
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

        with open(download_file_path, "wb") as file:
            file.write(file_content)

    def download_blobs_in_container(self, prefix):
        if self.local_blob_path is None:
            raise ValueError('Download path should be not none')
        blob_list = self.container_client.list_blobs()
        to_download = []
        for blob in blob_list:
            if str(blob.name).startswith(prefix):
                to_download.append(blob)
        for blob in tqdm(to_download, desc='downloading dataset'):
            # TODO: download by chunks to visualize
            # TODO: cache chunks to disk
            content = self.container_client.get_blob_client(blob).download_blob().readall()
            self.__save_blob__(blob.name, content)


def download_azure(directory=None, raw_data=False):
    logger.info(f'downloading data into {directory}')
    downloader = BlobFileDownloader(directory)
    if raw_data:
        prefix = 'raw'
    else:
        prefix = 'train'
    downloader.download_blobs_in_container(prefix=prefix)
    logger.info('Extracting files...')
    path = os.path.join(directory, f'{prefix}.tar.gz')
    with tarfile.open(path, mode="r:*") as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path=directory)
    return directory