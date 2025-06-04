import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile

from .env import KAGGLE_INPUT_PATH

# Number of bytes to read in each chunk when downloading
CHUNK_SIZE = 40960

# Mapping of “directory_name:URL(encoded)” pairs, separated by commas
DATA_SOURCE_MAPPING = (
    'celeba-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F29561%2F37705%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240503%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240503T211638Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D44933434f7f6976a174955c00dc35e1380278abc2ca6e7197e6092fe672ce8a4de6f84d25a8f4367e22d956c6821c8e4405d2ed5ab21f120d0d30b86e4fd8590b31a1f7de480ee757444febea3bf78d3c759017837e1313c7c9969a58678ed153acb3eb52faab2c1555a08169ce2f50725d962fcf9796b4714e25ea5e4e5faba6b05a55f7edae4ac99e84fa0b99b64c97929880ce2b05458fd2344332357751a04c42dd91ab715effcd868cfd6b03508ed989692b155c3f76296a1592d07a38bc9d5e7df95530ffdcbfb999c6fac5e3eab9a03d7feed25b15cd0d4fae7befcd62596302ce83e17b3f1291757daa8064477a891670a7c29b45a82a70bc99a4bd2,'
    'celeba-small-images-dataset:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F2078533%2F3450962%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240503%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240503T211638Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D78725995b0b743cd9b0cbcc6ff9d5f80d9d1cec33aec297800b6f6bf132e3fe937e833192a78a8ebb240ee18565e5ffc990abde7158cd1fb2d5a8ec2ef86256d4b7f07314d70dd51e6091e62f451802ce10bb066210cdfe2c0e859e51ce6a25a82ca1ed806164a799de72a95037c1ef428db6be616036f3a7622ed4464dc8280ec4b3cb5e92e29d27e5535c8de707a99a53e2139daeac8141df2f9c42c109478a64c0ec8ec89185a1780877ad25047a301a3fb86764911ea6500a2514f274a71963a220727b59a05621e361f7d6c28fc5369899bc4d026a76688721f55f907294fc94ac6e3eae3b5ba555ba1f4eb7e4619514fc12e5b99733611efd988c7d632,'
    'vgg19-weights:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F4827773%2F8160279%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240503%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240503T211638Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D21093dc203458d7d7cfdb439d1a253f612dc7a814b80337d27244a8a76850987787b3838df5af8816bcf1f0eb9f930e8dc87792d43addf755adc80d0d1261cfe75d5b72ba76263273af9d8ac95e19b3b3e7d64c065e1d6cab2e88f32a560eb581b5178759902f2ae0933f39c3e3b0c3476c59e6c527b4f5eb043f14e63cde3f8f514cc7a0d4e85151a8d570ca809a980ef07760eab21b51bb1fb1e365a2eb66dacc33c2b320b0361c63d1e2b941e5d9cb43dd63facfc799eb9d2e04bcdc56da3223f9daa750213451cae704b187cab0c089588302dd7772fbceed17d6d3921f3afe5bcd7aa561eb3989e33070f6ec5a44718cc8eb3596e583d951eba4533c8eb'
)


def download_data_sources():
    """
    Iterates over DATA_SOURCE_MAPPING, downloads each archive,
    shows a progress bar, then unzips (or untars) into /kaggle/input/<directory>.
    """
    for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
        directory, download_url_encoded = data_source_mapping.split(':')
        download_url = unquote(download_url_encoded)
        filename = urlparse(download_url).path  # to detect .zip vs .tar.*
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)

        try:
            # Open URL and write to a temporary file in chunks
            with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                total_length = fileres.headers.get('content-length')
                print(f'Downloading {directory}, {total_length} bytes compressed')
                downloaded = 0
                chunk = fileres.read(CHUNK_SIZE)
                while chunk:
                    downloaded += len(chunk)
                    tfile.write(chunk)
                    done = int(50 * downloaded / int(total_length))
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded} bytes downloaded")
                    sys.stdout.flush()
                    chunk = fileres.read(CHUNK_SIZE)

                # Extract based on file extension
                if filename.endswith('.zip'):
                    with ZipFile(tfile) as zf:
                        zf.extractall(destination_path)
                else:
                    with tarfile.open(tfile.name) as tf:
                        tf.extractall(destination_path)

                print(f'\nDownloaded and uncompressed: {directory}')

        except HTTPError:
            print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            continue
        except OSError:
            print(f'Failed to load {download_url} to path {destination_path}')
            continue

    print('Data source import complete.')
