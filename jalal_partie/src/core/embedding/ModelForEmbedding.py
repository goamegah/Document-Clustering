import multiprocessing
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from src.core.globals import DEFAULT_PATH_GLOVE_MODEL

class ModelForEmbedding:

    def __init__(self, sentences: list[str],vocab:dict[str,int]):
        self.sentences = sentences
        self.vocab=vocab
        if not isinstance(sentences, list):
            raise TypeError(f"{hex(id(self.sentences))} should be a list")
        for sentence in sentences:
            if not isinstance(sentence, str):
                raise TypeError(f"All elements of {hex(id(self.sentences))} should be a str")

    def word_embeddings(self, method="Word2vec", file_in=None, file_out=None,
                        min_count=5, workers=multiprocessing.cpu_count(),
                        vector_size=100, window=5) -> tuple[dict,dict]:
        if method == "Word2vec":
            model = self.generate_word2vec_model(
                min_count=min_count,
                workers=workers,
                vector_size=vector_size,
                window=window).wv
        elif method == "GloVe":
            if file_out == None:
                self.create_file_word2vec_toglv(file_in=file_in, file_out=file_out)
            model = self.generate_glove_modelfrom2vec(file_out)  # keyed vectors object
        vocab = [w for w in self.vocab if w in model.key_to_index.keys()]
        mapping={w:i for i,w in enumerate(vocab)}
        return {w:model[w] for w in vocab},mapping

    def generate_word2vec_model(self, min_count=5, workers=multiprocessing.cpu_count(), vector_size=100,
                                window=5) -> Word2Vec:
        s_tokens = [sentence.split() for sentence in self.sentences]
        return Word2Vec(s_tokens, min_count=min_count,
                        workers=workers,
                        vector_size=vector_size,
                        window=window
                        )

    def generate_glove_modelfrom2vec(self, file_out_name=None) -> KeyedVectors:
        return KeyedVectors.load_word2vec_format(file_out_name)

    def create_file_word2vec_toglv(self, file_in=None, file_out=None) -> str:
        glove_file = datapath(file_in)
        file_out_name = get_tmpfile(file_out)
        _ = glove2word2vec(glove_file, file_out_name)
        return file_out_name
