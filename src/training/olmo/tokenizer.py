import sentencepiece as spm


class Tokenizer:
    def __init__(self, filepath: str):
        self.sp = spm.SentencePieceProcessor(model_file=filepath)
        self.bos = 1
        self.eos = 2
        self.pad = 3

    def encode(self, text, out_type, add_bos=False, add_eos=False):
        return self.sp.encode(text, out_type=out_type, add_bos=add_bos, add_eos=add_eos)

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        return text

    def id_to_piece(self, id):
        return self.sp.id_to_piece(id)
    
    def piece_to_id(self, piece):
        return self.sp.piece_to_id(piece)
