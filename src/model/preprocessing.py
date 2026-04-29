import os
import re
import pandas as pd


class Preprocessor:
    def __init__(self):
        self.stopwords = []
        with open(os.environ.get("PARRENT_PATH")+'/dataset/stopword_id.txt', 'r') as r:
            self.stopwords = [sw.strip() for sw in r.readlines()]

    def preprocess_batch(self, dataframe: pd.DataFrame):
        """
        Deskripsi:
            Fungsi untuk melakukan preprocessing terhadap objek pd.Dataframe
        Input:
            - dataframe (pd.Dataframe): Objek dataframe yang berisikan file yang akan di olah
        Output:
            - clean_dataframe (pd.Dataframe): Objek dataframe yang berisikan file yang telah
            di olah
        """
        dataframe['Tweet'] = dataframe['Tweet'].map(self.preprocess)
        return dataframe

    def preprocess(self, text: str):
        """
        Deskripsi:
            Fungsi untuk melakukan preprocessing terhadap suatu tweet/teks
        Input:
            - text(str): Teks/tweet
        Output:
            - clean_text(str): Hasil teks yang telah di preproses
        """
        text = self._translate_username(text)
        text = self._translate_url(text)
        text = self._remove_punctuation(text)
        text = self._remove_stopword(text)
        return text


    def _translate_username(self, text):
        """
        Deskripsi:
            Fungsi untuk mengganti username dengan <USER_MENTION>
        Input:
            - text(str): Teks/tweet
        Output:
            - text(str): Teks yang sudah diganti username-nya
        """
        return re.sub(r'@\w+', '<USER_MENTION>', text)

    def _translate_url(self, text):
        """
        Deskripsi:
            Fungsi untuk mengganti URL dengan <URL>
        Input:
            - text(str): Teks/tweet
        Output:
            - text(str): Teks yang sudah diganti URL-nya
        """
        return re.sub(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])', '<URL>', text)

    def _remove_punctuation(self, text):
        """
        Deskripsi:
            Fungsi untuk menghilangkan tanda baca dengan spasi
        Input:
            - text(str): Teks/tweet
        Output:
            - text(str): Teks yang sudah dihilangkan tanda bacanya
        """
        text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-`~()]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text

    def _remove_stopword(self, text):
        """
        Deskripsi:
            Fungsi untuk menghilangkan kata-kata yang tidak bermakna (stopword)
        Input:
            - text(str): Teks/tweet
        Output:
            - text(str): Teks yang sudah dihilangkan kata-kata yang tidak bermakna sentiment
        """
        text = text.split()
        text = [word for word in text if len(word) > 2]
        text = ' '.join([word.lower() for word in text if word not in self.stopwords])
        return text
