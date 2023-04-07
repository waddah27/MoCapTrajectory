class CFG:
    def __init__(
            self,
            dim_model:int = 512,
            dim_feedforward:int = 2048,
            n_heads:int = 8,
            n_encoder_layers:int = 6,
            n_decoder_layers:int = 6,
            dropout:float = 0.1,
             max_seq_len = 5000 ) -> None:
        self.dim_model = dim_model
        self.dim_feedforward = dim_feedforward
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        