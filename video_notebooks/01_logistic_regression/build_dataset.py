from dataset import Dataset
from email import Email
import os
import os.path
import pickle
from word_encoding_dictionary import WordEncodingDictionary

DATA_DIR = os.path.join(
    os.getcwd(),
    "data/"
)
ENRON_DATA_DIR_NAME = "enron1"

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

    return Dataset(
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

build_and_save_dataset()
