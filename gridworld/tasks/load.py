import requests
from tqdm import tqdm


def download(url, destination, data_prefix):
    r = requests.get(url, stream=True)
    CHUNK_SIZE = 1024
    total_length = int(r.headers.get('content-length'))
    with open(destination, "wb") as f:
        with tqdm(desc=f'downloading task dataset into {data_prefix}', 
                  total=(total_length // 1024) + 1) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(CHUNK_SIZE // CHUNK_SIZE)
