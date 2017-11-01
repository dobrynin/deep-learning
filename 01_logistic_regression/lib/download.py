from lib.config import DATA_DIR
from lib.dataset import RawDataset
from lib.email import Email
from lib.word_encoding_dictionary import WordEncodingDictionary
import os
import os.path
import pickle
from urllib.request import urlretrieve

ENRON_SPAM_URL = (
    "http://csmining.org/index.php/"
    "enron-spam-datasets.html"
    "?file=tl_files/Project_Datasets/Enron-Spam%20datasets/Preprocessed"
    "/enron1.tar.tar"
)

TAR_FILE_NAME = "enron1.tar.tar"
ENRON_DATA_DIR_NAME = "enron1"

def download_tarfile():
    tarfile_path = os.path.join(DATA_DIR, TAR_FILE_NAME)
    if os.path.isfile(tarfile_path):
        print("Tarfile already downloaded!")
        return

    print("Downloading enron1.tar.tar")
    urlretrieve(ENRON_SPAM_URL, tarfile_path)
    print("Download complete!")

def extract_tarfile():
    tarfile_path = os.path.join(DATA_DIR, TAR_FILE_NAME)
    enron_data_dir = os.path.join(DATA_DIR, ENRON_DATA_DIR_NAME)
    if os.path.isdir(enron_data_dir):
        print("Tarfile already extracted!")
        return

    print("Extracting enron1.tar.tar")
    os.system(f"tar -xf {tarfile_path} -C {DATA_DIR}")
    print("Extraction complete!")

def read_emails_dir(word_encoding_dictionary, path, label):
    emails = []
    for email_fname in os.listdir(os.path.join(DATA_DIR, path)):
        email_path = os.path.join(path, email_fname)
        email = Email.read(
            path = email_path,
            word_encoding_dictionary = word_encoding_dictionary,
            label = label
        )
        emails.append(email)

    return emails

def build_dataset():
    word_encoding_dictionary = WordEncodingDictionary()
    ham_emails = read_emails_dir(
        word_encoding_dictionary = word_encoding_dictionary,
        path = os.path.join(ENRON_DATA_DIR_NAME, "ham"),
        label = 0
    )
    spam_emails = read_emails_dir(
        word_encoding_dictionary = word_encoding_dictionary,
        path = os.path.join(ENRON_DATA_DIR_NAME, "spam"),
        label = 1
    )

    return RawDataset(
        word_encoding_dictionary = word_encoding_dictionary,
        ham_emails = ham_emails,
        spam_emails = spam_emails
    )

def save_dataset(dataset):
    with open("data/data.p", "wb") as f:
        pickle.dump(dataset, f)

def build_and_save_dataset():
    if os.path.isfile("data/data.p"):
        print("Dataset already processed!")
        return

    print("Reading and processing emails!")
    dataset = build_dataset()
    save_dataset(dataset)
    print("Dataset created!")

def run():
    download_tarfile()
    extract_tarfile()
    build_and_save_dataset()
